from Crop_Image_Classification.config.configuration import ConfigurationManager
from Crop_Image_Classification.components.evaluation import Evaluation
from Crop_Image_Classification import logger




STAGE_NAME_EVALUATION = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(val_config)
        evaluation.evaluation()
        evaluation.save_score()



if __name__ == '__main__':
    try:
        logger.info(f'\n\n{"**"*50}\nStarted {STAGE_NAME_EVALUATION}\n{"**"*50}\n')
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f'\n\n{"**"*50}\nCompleted {STAGE_NAME_EVALUATION}\n{"**"*50}\n\n')
    except Exception as e:
        logger.exception(e)
        raise e
            