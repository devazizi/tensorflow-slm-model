
def get_model_embeddings(db):
    query = "SELECT * FROM model_embeddings"
    result = db.query(query)
    return result.result_rows


def get_predicts(db):
    query_predictions = "SELECT * FROM model_predictions"
    result_predictions = db.query(query_predictions)
    return result_predictions.result_rows