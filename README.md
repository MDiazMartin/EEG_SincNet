# EEG_SincNet
Título del proyecto: Interfaces cerebro-ordenador basadas en EEG para la síntesis de voz

El objetivo principal del proyecto es demostrar la viabilidad del diseño de un sistema de conversión de señales de EEG obtenidas por medios invasivos (electrodos profundos) a voz.

La carpeta 'Redes neuronales' contiene todos los modelos de redes neuronales diseñadas para los datos de EEG y para los datos de PMA, además de scripts que hacen uso del vocoder WORLD para la parametrización y reconstrucción de las señales de voz. Concretamente hay dos tipos de redes neuronales: una red neuronal basada en SincNet (propósito del proyecto) y otra red DNN simple utilizada para comparar resultados.

La carpeta 'SingleWordProductionDutch-main' contiene los scripts utilizados para la extracción de características high-hamma necesarios para extraer previamente dichas características y poder procesar las señales con la DNN estándar (la red SincNet propuesta procesa las señales de EEG en crudo).
