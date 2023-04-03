from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

def run_test():
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transform(train_data, test_data)

    model_trainer = ModelTrainer()
    try:
        best_model, best_model_name, best_model_score = model_trainer.initiate_model_trainer(train_arr=train_arr, test_arr=test_arr)
        print(f'Best Model score is {best_model_name} with score {best_model_score}')
    except Exception as e:
        raise e




if __name__ == "__main__":
    run_test()