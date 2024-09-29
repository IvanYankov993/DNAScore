
def evaluate_model(model, X_test, y_test):
    results = model.evaluate(X_test, y_test)
    print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")
    return results
