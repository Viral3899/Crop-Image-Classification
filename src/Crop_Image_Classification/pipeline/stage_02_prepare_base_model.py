from Crop_Image_Classification.components.prepare_base_model import PrepareBaseModel
from Crop_Image_Classification.config.configuration import ConfigurationManager
from Crop_Image_Classification import logger
 
 
STAGE_NAME_PREPARE_BASE_MODEL = 'Prepare Base Model'

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            config = ConfigurationManager()
            prepare_base_model_config = config.get_prepare_base_model_config()
            prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
            prepare_base_model.get_base_model()
            prepare_base_model.update_base_model()
        except Exception as e:
            raise e
        
        
if __name__ == '__main__':
    
    try:
        logger.info(f'\n\n{"**"*50}\nStarted {STAGE_NAME_PREPARE_BASE_MODEL}\n{"**"*50}\n')
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f'\n\n{"**"*50}\nCompleted {STAGE_NAME_PREPARE_BASE_MODEL}\n{"**"*50}\n\n')
        
    except Exception as e:
        logger.exception(e)
        raise e