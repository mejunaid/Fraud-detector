from src.preprocess import load_data, preprocess_data
from src.model import train_model, evaluate_model
from src.utils import plot_confusion, plot_roc_curve

def run_pipeline():
    df = load_data('data/creditcard.csv')
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    model = train_model(X_train, y_train)
    y_pred, y_proba = evaluate_model(model, X_test, y_test)
    
    plot_confusion(y_test, y_pred)
    plot_roc_curve(model, X_test, y_test)

if __name__ == "__main__":
    run_pipeline()
