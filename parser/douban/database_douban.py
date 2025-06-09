# database_douban.py

import pymysql

# --- 数据库配置 ---
DB_CONFIG = {
    'host': '127.0.0.1', 'port': 3306, 'user': 'root',
    'password': '',  # <-- 请在这里填入您的MySQL密码
    'database': 'bigwork2025',
    'charset': 'utf8mb4', 'cursorclass': pymysql.cursors.DictCursor
}


def get_db_connection():
    """建立数据库连接"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        print("✔️  MySQL数据库连接成功！")
        return connection
    except pymysql.MySQLError as e:
        print(f"❌  MySQL数据库连接失败: {e}")
        return None


def _upsert_entity(cursor, entity_name: str, table_name: str) -> int:
    """通用函数：插入或获取一个实体（如导演、演员）的ID。"""
    cursor.execute(f"SELECT id FROM `{table_name}` WHERE name = %s", (entity_name,))
    result = cursor.fetchone()
    if result:
        return result['id']
    try:
        cursor.execute(f"INSERT INTO `{table_name}` (name) VALUES (%s)", (entity_name,))
        return cursor.lastrowid
    except pymysql.err.IntegrityError:
        cursor.execute(f"SELECT id FROM `{table_name}` WHERE name = %s", (entity_name,))
        return cursor.fetchone()['id']


def _link_movie_to_entities(cursor, movie_id: int, entity_ids: list, link_table: str, movie_col: str, entity_col: str):
    """通用函数：建立电影与多个实体的关联关系"""
    if not entity_ids: return
    cursor.execute(f"DELETE FROM `{link_table}` WHERE `{movie_col}` = %s", (movie_id,))
    sql = f"INSERT IGNORE INTO `{link_table}` ({movie_col}, {entity_col}) VALUES (%s, %s)"
    data = [(movie_id, entity_id) for entity_id in entity_ids]
    cursor.executemany(sql, data)


def store_douban_movie(connection, movie_data: dict):
    """存储单条豆瓣Top250电影数据，并处理所有关联信息。"""
    with connection.cursor() as cursor:
        try:
            # 将genres列表转换为字符串，以便存入主表
            movie_data['genres_text'] = ' / '.join(movie_data.get('genres', []))

            # SQL语句包含所有douban_top250表的字段
            sql_upsert = """
            INSERT INTO douban_top250 (
                ranking, douban_id, name_cn, name_en, score, comment_count, genres_text, quote,
                release_date_text, `year`, runtime_text, synopsis, cover_url, douban_url, hot_comments_json
            ) VALUES (
                %(ranking)s, %(douban_id)s, %(name_cn)s, %(name_en)s, %(score)s, %(comment_count)s, %(genres_text)s, %(quote)s,
                %(release_date_text)s, %(year)s, %(runtime_text)s, %(synopsis)s, %(cover_url)s, %(douban_url)s, %(comments_json)s
            ) ON DUPLICATE KEY UPDATE
                ranking=VALUES(ranking), name_cn=VALUES(name_cn), name_en=VALUES(name_en), score=VALUES(score),
                comment_count=VALUES(comment_count), genres_text=VALUES(genres_text), quote=VALUES(quote),
                release_date_text=VALUES(release_date_text), `year`=VALUES(`year`), runtime_text=VALUES(runtime_text),
                synopsis=VALUES(synopsis), cover_url=VALUES(cover_url),
                hot_comments_json=VALUES(hot_comments_json), updated_at=NOW();
            """
            cursor.execute(sql_upsert, movie_data)

            # 获取电影ID用于关联表操作
            cursor.execute("SELECT id FROM douban_top250 WHERE douban_id = %s", (movie_data['douban_id'],))
            movie_id = cursor.fetchone()['id']

            # 定义实体与关联表的映射关系
            entity_map = {
                'directors': ('douban_top250_directors', 'director_id'),
                'actors': ('douban_top250_actors', 'actor_id'),
                'genres': ('douban_top250_genres', 'genre_id'),
                'regions': ('douban_top250_regions', 'region_id'),
                'languages': ('douban_top250_languages', 'language_id')
            }

            for key, (link_table, id_col) in entity_map.items():
                names = movie_data.get(key, [])
                if names:
                    entity_table_name = 'genres' if key == 'genres' else key  # genres实体表就叫genres
                    ids = [_upsert_entity(cursor, name, entity_table_name) for name in names]
                    _link_movie_to_entities(cursor, movie_id, ids, link_table, 'douban_movie_id', id_col)

            connection.commit()
            print(f"  💾  Top250电影 '{movie_data['name_cn']}' 数据已存入/更新。")
        except Exception as e:
            connection.rollback()
            import traceback
            print(f"  ❌  Top250电影 '{movie_data['name_cn']}' 数据库操作失败: {e}\n{traceback.format_exc()}")