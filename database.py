import warnings
import pymysql
import pandas as pd
import os

# 定义数据库连接
# 注意：在实际部署中，不建议将敏感信息（如密码）硬编码在代码中
# 建议使用环境变量或配置文件来管理数据库凭据
db = pymysql.connect(host='localhost',
                     user='root',
                     password='',  # XAMPP默认root用户密码为空
                     database='douban',
                     charset='utf8mb4')  # 确保字符集支持中文和特殊符号

# 定义游标
cursor = db.cursor()

def _insert_entity_and_link(movie_id: int, entities: list, entity_table_name: str, link_table_name: str, movie_id_col_name: str):
    """
    辅助函数：插入实体（如导演、演员、地区等）到其各自的表中，
    并建立与电影（豆瓣或猫眼）的关联关系。

    参数:
        movie_id (int): 当前电影在主表（movie或maoyan_movie）中的ID。
        entities (list): 实体名称列表（如 ['导演A', '导演B']）。
        entity_table_name (str): 实体表名（如 'director', 'actor', 'place', 'lang', 'type'）。
        link_table_name (str): 关联表名（如 'movie_director', 'maoyan_movie_actor'）。
        movie_id_col_name (str): 关联表中表示电影ID的列名（'movie_id' 或 'maoyan_id'）。
    """
    if not entities:
        return

    for entity_name in entities:
        # 1. 检查实体是否已存在
        sql_select = f'SELECT id FROM `{entity_table_name}` WHERE name=%s'
        cursor.execute(sql_select, (entity_name,))
        res = cursor.fetchone() # fetchone更高效，因为只需要一个结果

        entity_id = None
        if res:
            entity_id = res[0] # 实体已存在，获取其ID
        else:
            # 2. 如果实体不存在，则插入新实体
            sql_insert = f'INSERT INTO `{entity_table_name}`(name) VALUES (%s)'
            cursor.execute(sql_insert, (entity_name,))
            entity_id = cursor.lastrowid # 获取新插入的ID

        # 3. 建立电影与实体的关联关系
        sql_insert_link = f'INSERT INTO `{link_table_name}`(`{entity_table_name}_id`, `{movie_id_col_name}`) VALUES(%s, %s)'
        try:
            cursor.execute(sql_insert_link, (entity_id, movie_id))
        except pymysql.err.IntegrityError:
            # 捕获IntegrityError（如Duplicate entry for primary key），表示关联已存在，跳过
            pass
        except Exception as e:
            print(f"Error inserting into {link_table_name} for {entity_table_name} '{entity_name}': {e}")


def db_store(movie_data: list):
    """
    存储豆瓣电影数据及其关联信息到数据库。
    """
    # 解包 movie_data，确保顺序与main.py中get_data函数的movie_data列表顺序一致
    # movie_data = [name1, name2, score, comment, quote_str, page_url, director, actor, movie_type, place, lang, year, length]
    name1, name2, score, comment, quote_str, page_url, director, actor, movie_type, place, lang, year, length = movie_data

    # 将列表类型的数据转换为逗号分隔的字符串，以便存储到主表的VARCHAR/TEXT字段
    director_str = ','.join(director) if isinstance(director, list) else str(director)
    actor_str = ','.join(actor) if isinstance(actor, list) else str(actor)
    type_str = ','.join(movie_type) if isinstance(movie_type, list) else str(movie_type)
    place_str = ','.join(place) if isinstance(place, list) else str(place)
    lang_str = ','.join(lang) if isinstance(lang, list) else str(lang)

    # 插入movie表的主SQL语句
    sql_movie_insert = """
    INSERT INTO movie (
        `中文名`, `外文名`, `评分`, `评价人数`, `电影语录`, `详情URL`,
        `导演`, `主演`, `类型`, `地区`, `语言`, `上映年份`, `时长`
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    values_movie = (
        name1, name2, score, comment, quote_str, page_url,
        director_str, actor_str, type_str, place_str, lang_str, year, length
    )

    try:
        # 执行主表插入
        cursor.execute(sql_movie_insert, values_movie)
        movie_id = cursor.lastrowid # 获取新插入电影的ID

        # 使用辅助函数插入关联数据
        _insert_entity_and_link(movie_id, director, 'director', 'movie_director', 'movie_id')
        _insert_entity_and_link(movie_id, actor, 'actor', 'movie_actor', 'movie_id')
        _insert_entity_and_link(movie_id, place, 'place', 'movie_place', 'movie_id')
        _insert_entity_and_link(movie_id, lang, 'lang', 'movie_lang', 'movie_id')
        _insert_entity_and_link(movie_id, movie_type, 'type', 'movie_type', 'movie_id') # 注意这里传入的是 movie_type 变量

        db.commit()  # 在所有相关数据插入完成后再统一提交
        print(f"豆瓣电影 '{name1}' 及其关联信息插入成功！")

    except Exception as e:
        db.rollback()  # 发生错误时回滚所有操作
        print(f"豆瓣电影 '{name1}' 数据存储失败: {e}")
        # print(f"SQL for movie insert: {sql_movie_insert}") # 如果需要调试可以取消注释
        # print(f"Values for movie insert: {values_movie}")


def db_store_2(data: list):
    """
    存储猫眼电影数据及其关联信息到数据库。
    """
    # 解包 data，根据 main.py 中 get_data_maoyan 的 DataFrame 列名推断
    # data = ['电影名', '上映日期', '票房(万元)', '平均票价', '场均人数', '豆瓣评分', '豆瓣评论数', '导演', '演员', '类型', '地区', '语言', '时长']
    name, date, money, avg_money, avg_people, score, comment, director, actor, movie_type, place, lang, length = data

    # 同样将列表转换为字符串存储到主表 (maoyan_movie)
    director_str = ','.join(director) if isinstance(director, list) else str(director)
    actor_str = ','.join(actor) if isinstance(actor, list) else str(actor)
    type_str = ','.join(movie_type) if isinstance(movie_type, list) else str(movie_type)
    place_str = ','.join(place) if isinstance(place, list) else str(place)
    lang_str = ','.join(lang) if isinstance(lang, list) else str(lang)

    # 插入 maoyan_movie 表的主SQL语句，使用参数化查询
    sql_maoyan_insert = """
    INSERT INTO maoyan_movie (
        `电影名`, `上映日期`, `票房(万元)`, `平均票价`, `场均人数`, `豆瓣评分`, `豆瓣评论数`,
        `导演`, `演员`, `类型`, `地区`, `语言`, `时长`
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values_maoyan = (
        name, date, money, avg_money, avg_people, score, comment,
        director_str, actor_str, type_str, place_str, lang_str, length
    )

    try:
        cursor.execute(sql_maoyan_insert, values_maoyan)
        maoyan_id = cursor.lastrowid  # 获取猫眼电影的ID

        # 使用辅助函数插入关联数据
        _insert_entity_and_link(maoyan_id, director, 'director', 'maoyan_movie_director', 'maoyan_id')
        _insert_entity_and_link(maoyan_id, actor, 'actor', 'maoyan_movie_actor', 'maoyan_id')
        _insert_entity_and_link(maoyan_id, place, 'place', 'maoyan_movie_place', 'maoyan_id')
        _insert_entity_and_link(maoyan_id, lang, 'lang', 'maoyan_movie_lang', 'maoyan_id')
        _insert_entity_and_link(maoyan_id, movie_type, 'type', 'maoyan_movie_type', 'maoyan_id') # 注意这里传入的是 movie_type 变量

        db.commit()  # 在所有相关数据插入完成后再统一提交
        print(f"猫眼电影 '{name}' 及其关联信息插入成功！")

    except Exception as e:  # 捕获整个db_store_2函数的异常
        db.rollback()  # 发生任何错误时回滚所有操作
        print(f"猫眼电影 '{name}' 数据存储失败: {e}")
        # print(f"SQL for maoyan_movie insert: {sql_maoyan_insert}") # 如果需要调试可以取消注释
        # print(f"Values for maoyan_movie insert: {values_maoyan}")


def get_data_from_res(res, params):
    """
    从数据库查询结果中读取参数列。
    （此函数在当前 main.py 中似乎未直接使用，但保留其原有功能）
    """
    for row in res:
        for j in range(len(params)):
            params[j].append(row[j])


def get_table_names() -> list[str]:
    """
    读取数据库中所有表的名称。
    """
    sql = 'select table_name from information_schema.tables where table_schema="douban"'
    cursor.execute(sql)
    results = cursor.fetchall()
    names = [row[0] for row in results]  # 提取表名
    return names


def csv_store():
    """
    读取数据库内容到DataFrame，并存到CSV文件。
    为所有数据库表创建CSV文件。
    """
    # 确保 ./csv 目录存在，如果不存在则创建
    csv_dir = "./csv"
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
        print(f"Created directory: {csv_dir}")

    names = get_table_names()
    if not names:
        print("Warning: No tables found in 'douban' database to export to CSV.")
        return

    print("\n--- Exporting database tables to CSVs ---")
    for name in names:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # 忽略PyMySQL的警告
            try:
                # 使用f-string和反引号确保表名正确，防止SQL注入或特殊字符问题
                df = pd.read_sql(f'SELECT * FROM `{name}`', db)
                # index=False 避免将DataFrame索引写入CSV
                # encoding='utf-8-sig' (带BOM的UTF-8) 确保Excel等软件能正确识别中文
                csv_file_path = os.path.join(csv_dir, f'{name}.csv')
                df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
                print(f"Table '{name}' exported to {os.path.basename(csv_file_path)} successfully.")
            except Exception as e:
                print(f"Error exporting table '{name}' to CSV: {e}")
    print("--- Database tables export to CSVs finished ---\n")


def execute_sql(sql: str, param_num: int) -> list[list]:
    """
    运行SQL，返回指定数目的参数列表。
    （此函数在当前 main.py 中似乎未直接使用，但保留其原有功能）
    """
    cursor.execute(sql)
    params = [[] for _ in range(param_num)]  # 使用 _ 作为循环变量

    results = cursor.fetchall()  # 获取所有结果

    # 填充 params 列表
    for row in results:
        # 确保不越界，且只取需要的列
        for j in range(min(param_num, len(row))):
            params[j].append(row[j])

    return params