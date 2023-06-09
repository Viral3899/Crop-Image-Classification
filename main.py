from Crop_Image_Classification import logger
from Crop_Image_Classification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Crop_Image_Classification.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from Crop_Image_Classification.pipeline.stage_03_training import ModelTrainingPipeline
from Crop_Image_Classification.pipeline.stage_04_evaluation import EvaluationPipeline


STAGE_NAME_INGESTION = 'Data Ingestion Stage'

try:
    logger.info(f'\n\n{"**"*50}\nStarted {STAGE_NAME_INGESTION}\n{"**"*50}\n')
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f'\n\n{"**"*50}\nCompleted {STAGE_NAME_INGESTION}\n{"**"*50}\n\n')
    
except Exception as e:
    logger.exception(e)
    raise e


# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = DataIngestionTrainingPipeline()
#    data_ingestion.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e


STAGE_NAME_PREPARE_BASE_MODEL = 'Prepare Base Model'

try:
    logger.info(f'\n\n{"**"*50}\nStarted {STAGE_NAME_PREPARE_BASE_MODEL}\n{"**"*50}\n')
    prepare_model = PrepareBaseModelTrainingPipeline()
    prepare_model.main()
    logger.info(f'\n\n{"**"*50}\nCompleted {STAGE_NAME_PREPARE_BASE_MODEL}\n{"**"*50}\n\n')
    
except Exception as e:
    logger.exception(e)
    raise e
 
STAGE_NAME_TRAINING= 'Training'
try:
    logger.info(f'\n\n{"**"*50}\nStarted {STAGE_NAME_TRAINING}\n{"**"*50}\n')
    trainer = ModelTrainingPipeline()
    trainer.main()
    logger.info(f'\n\n{"**"*50}\nCompleted {STAGE_NAME_TRAINING}\n{"**"*50}\n\n')
    
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME_EVALUATION = "Evaluation stage"

try:
    logger.info(f'\n\n{"**"*50}\nStarted {STAGE_NAME_EVALUATION}\n{"**"*50}\n')
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f'\n\n{"**"*50}\nCompleted {STAGE_NAME_EVALUATION}\n{"**"*50}\n\n')
except Exception as e:
    logger.exception(e)
    raise e