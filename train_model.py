from ml_model import train_model


if __name__ == "__main__":
    metrics = train_model()
    print(f"Model trained with accuracy: {metrics['accuracy']:.2%}")
    print(f"Training rows: {metrics['training_rows']}")
    print(f"Test rows: {metrics['test_rows']}")
