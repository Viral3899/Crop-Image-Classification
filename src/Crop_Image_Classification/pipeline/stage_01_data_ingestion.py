from Crop_Image_Classification.components.data_ingestion import DataIngestion
from Crop_Image_Classification.config.configuration import ConfigurationManager
from Crop_Image_Classification import logger
 
 
STAGE_NAME_INGESTION = 'Data Ingestion Stage'

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        
        
        
if __name__ == '__main__':
    
    try:
        logger.info(f'\n\n{"**"*50}\nStarted {STAGE_NAME_INGESTION}\n{"**"*50}\n')
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f'\n\n{"**"*50}\nCompleted {STAGE_NAME_INGESTION}\n{"**"*50}\n\n')
        
    except Exception as e:
        logger.exception(e)
        raise e