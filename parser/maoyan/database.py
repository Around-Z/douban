# database.py

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
    """插入或获取一个实体（如导演、演员）的ID。"""
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
    """建立电影与多个实体的关联关系"""
    if not entity_ids: return
    cursor.execute(f"DELETE FROM `{link_table}` WHERE `{movie_col}` = %s", (movie_id,))
    sql = f"INSERT IGNORE INTO `{link_table}` ({movie_col}, {entity_col}) VALUES (%s, %s)"
    data = [(movie_id, entity_id) for entity_id in entity_ids]
    cursor.executemany(sql, data)


def store_maoyan_movie(connection, movie_data: dict):
    """
    【最终版】存储猫眼电影及其所有关联信息，包括电影类型。
    """
    with connection.cursor() as cursor:
        try:
            # 【核心修复】在SQL语句中加入 genres_text
            # 将列表转换为字符串以便存入主表
            movie_data['genres_text'] = ' / '.join(movie_data.get('genres', []))

            sql_upsert = """
            INSERT INTO maoyan_movies (
                name, name_en, release_date, box_office_ten_thousand, avg_ticket_price, avg_audience_per_screening,
                douban_id, douban_score, douban_comment_count, genres_text, quote, release_date_text, `year`, runtime_text,
                synopsis, cover_url, douban_url, hot_comments_json
            ) VALUES (
                %(name)s, %(name_en)s, %(release_date)s, %(box_office)s, %(avg_price)s, %(avg_audience)s,
                %(douban_id)s, %(douban_score)s, %(douban_comment_count)s, %(genres_text)s, %(quote)s, %(release_date_text)s, %(year)s, %(runtime_text)s,
                %(synopsis)s, %(cover_url)s, %(douban_url)s, %(comments_json)s
            ) ON DUPLICATE KEY UPDATE
                name_en=VALUES(name_en), box_office_ten_thousand=VALUES(box_office_ten_thousand), avg_ticket_price=VALUES(avg_ticket_price),
                avg_audience_per_screening=VALUES(avg_audience_per_screening), douban_id=VALUES(douban_id),
                douban_score=VALUES(douban_score), douban_comment_count=VALUES(douban_comment_count), genres_text=VALUES(genres_text),
                quote=VALUES(quote), release_date_text=VALUES(release_date_text), `year`=VALUES(`year`), runtime_text=VALUES(runtime_text),
                synopsis=VALUES(synopsis), cover_url=VALUES(cover_url), douban_url=VALUES(douban_url),
                hot_comments_json=VALUES(hot_comments_json), updated_at=NOW();
            """
            cursor.execute(sql_upsert, movie_data)

            cursor.execute("SELECT id FROM maoyan_movies WHERE name = %s AND release_date = %s",
                           (movie_data['name'], movie_data['release_date']))
            result = cursor.fetchone()
            if not result: return

            maoyan_id = result['id']

            # 【核心修复】关联表逻辑现在包含了 genres
            entity_map = {
                'directors': ('maoyan_movie_directors', 'director_id'),
                'actors': ('maoyan_movie_actors', 'actor_id'),
                'genres': ('maoyan_movie_genres', 'genre_id')  # <-- 新增
            }
            for key, (link_table, id_col) in entity_map.items():
                names = movie_data.get(key, [])
                if names:
                    entity_table_name = 'genres' if key == 'genres' else key  # 确保使用正确的实体表名
                    ids = [_upsert_entity(cursor, name, entity_table_name) for name in names]
                    _link_movie_to_entities(cursor, maoyan_id, ids, link_table, 'maoyan_movie_id', id_col)

            connection.commit()
            print(f"  💾  猫眼电影 '{movie_data['name']}' 的完整数据已存入/更新。")
        except Exception as e:
            connection.rollback()
            print(f"  ❌  猫眼电影 '{movie_data['name']}' 数据库操作失败: {e}")