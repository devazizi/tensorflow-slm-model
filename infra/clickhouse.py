import clickhouse_connect


def get_clickhouse():
    return clickhouse_connect.get_client(
        host='localhost',
        port=8123,
        database='slm',
        username='clickhouse',
        password='clickhouse_password'
    )


def create_tables(db):
    db.command('''
    CREATE TABLE IF NOT EXISTS model_embeddings (
        id UInt32,
        embedding Array(Float32)
    ) ENGINE = MergeTree() ORDER BY id
    ''')

    db.command('''
    CREATE TABLE IF NOT EXISTS model_predictions (
        sequence Array(UInt32),
        predicted Array(Float32)
    ) ENGINE = MergeTree() ORDER BY sequence
    ''')

    db.command('''
    CREATE TABLE IF NOT EXISTS model_dense_weights (
        layer UInt32,
        weights Array(Float32)
    ) ENGINE = MergeTree() ORDER BY layer
    ''')

