import logging

def preprocess_data():
    logger = logging.getLogger(__name__)
    logger.debug("Iniciando preprocesado...")  
    # No se ha realizado preprocesado para MNIST, pues las imágenes ya vienen en un formato adecuado.
    # Sin embargo, si quisiéramos agregar algún tipo de preprocesado, se reservará este fichero para ello.
    logger.debug("Preprocesado completado correctamente")