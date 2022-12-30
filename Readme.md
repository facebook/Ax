
Facebook
/
Hacha
Público
Plataforma de Experimentación Adaptativa

hacha.dev
Licencia
 licencia MIT
 2k estrellas 238 tenedores 
Código
Cuestiones
17
Solicitudes de extracción
29
Comportamiento
Proyectos
Seguridad
Perspectivas
facebook/hacha
Última confirmación
@Balandat
@facebook-github-bot
Balandat y facebook-github-bot
…
10 hours ago
Estadísticas de Git
archivos
LÉAME.md
Logotipo de hacha

Apoya a Ucrania Estado de construcción Estado de construcción Estado de construcción Estado de construcción código cov Estado de construcción

Axe es una plataforma accesible de propósito general para comprender, administrar, implementar y automatizar experimentos adaptativos.

La experimentación adaptativa es el proceso guiado por aprendizaje automático de exploración iterativa de un espacio de parámetros (posiblemente infinito) para identificar configuraciones óptimas de una manera eficiente en recursos. Axe actualmente admite la optimización bayesiana y la optimización de bandidos como estrategias de exploración. La optimización bayesiana en Axe está impulsada por BoTorch , una biblioteca moderna para la investigación de optimización bayesiana basada en PyTorch.

Para ver la documentación y los tutoriales completos, consulte el sitio web de Axe

¿Por qué Ax?
Versatilidad : Axe admite diferentes tipos de experimentos, desde pruebas dinámicas A/B asistidas por ML hasta optimización de hiperparámetros en aprendizaje automático.
Personalización : Axe facilita la adición de nuevos modelos y algoritmos de decisión, lo que permite la investigación y el desarrollo con una sobrecarga mínima.
Producción completa : Axe viene con integración de almacenamiento y capacidad para guardar y recargar completamente los experimentos.
Soporte para experimentación multimodal y restringida : Axe permite ejecutar y combinar múltiples experimentos (p. ej., simulación con una prueba A/B "en línea" del mundo real) y optimización restringida (p. ej., mejorar la precisión de la clasificación sin un aumento significativo en la utilización de recursos) .
Eficiencia en entornos con mucho ruido : Axe ofrece algoritmos de última generación específicamente diseñados para experimentos ruidosos, como simulaciones con agentes de aprendizaje por refuerzo.
Facilidad de uso : Axe incluye 3 API diferentes que logran diferentes equilibrios entre estructura ligera y flexibilidad. Utilizando la API de bucle más concisa, se puede realizar una optimización completa en una sola llamada de función. La API de servicio se integra fácilmente con programadores externos. La API de desarrollador más elaborada permite la personalización completa del algoritmo y la introspección del experimento.
Empezando
Para ejecutar un bucle de optimización simple en Axe (utilizando la superficie de respuesta de Booth como función de evaluación artificial):

>>> from ax import optimize
>>> best_parameters, best_values, experiment, model = optimize(
        parameters=[
          {
            "name": "x1",
            "type": "range",
            "bounds": [-10.0, 10.0],
          },
          {
            "name": "x2",
            "type": "range",
            "bounds": [-10.0, 10.0],
          },
        ],
        # Booth function
        evaluation_function=lambda p: (p["x1"] + 2*p["x2"] - 7)**2 + (2*p["x1"] + p["x2"] - 5)**2,
        minimize=True,
    )

# best_parameters contains {'x1': 1.02, 'x2': 2.97}; the global min is (1, 3)
Instalación
Requisitos
Necesita Python 3.8 o posterior para ejecutar Axe.

Las dependencias requeridas de Python son:

botorch
jinja2
pandas
espía
aprender
gráficamente >=2.2.1
Versión estable
Instalación a través de pip
Recomendamos instalar Axe a través de pip (incluso si usa el entorno Conda):

conda install pytorch torchvision -c pytorch  # OSX only (details below)
pip install ax-platform
La instalación utilizará Python Wheels de PyPI, disponible para OSX, Linux y Windows .

Nota : asegúrese de que el que pipse está utilizando para instalar ax-platformsea realmente el del entorno Conda recién creado. Si está utilizando un sistema operativo basado en Unix, puede utilizar which pippara comprobar.

Recomendación para usuarios de MacOS : PyTorch es una dependencia requerida de BoTorch y se puede instalar automáticamente a través de pip. Sin embargo, le recomendamos que instale PyTorch manualmente antes de instalar Axe, utilizando el administrador de paquetes de Anaconda . La instalación desde Anaconda se vinculará con MKL (una biblioteca que optimiza el cálculo matemático para los procesadores Intel). Esto dará como resultado una aceleración de hasta un orden de magnitud para la optimización bayesiana, ya que en este momento, la instalación de PyTorch desde pip no se vincula con MKL.

Si necesita CUDA en MacOS, deberá compilar PyTorch desde la fuente. Consulte las instrucciones de instalación de PyTorch anteriores.

Dependencias opcionales
Para usar Axe con un entorno de notebook, necesitará Jupyter. Instalarlo primero:

pip install jupyter
Si desea almacenar los experimentos en MySQL, necesitará SQLAlchemy:

pip install SQLAlchemy
Ultima versión
Instalando desde Git
Puede instalar la última versión (de última generación) de Git:

pip install git+https://github.com/facebook/Ax.git#egg=ax-platform
Consulte la recomendación anterior para instalar PyTorch para usuarios de MacOS.

A veces, la vanguardia de Axe puede depender de las versiones de última generación de BoTorch (o GPyTorch). Por lo tanto, recomendamos instalar también los de Git:

pip install git+https://github.com/cornellius-gp/gpytorch.git
pip install git+https://github.com/pytorch/botorch.git
Dependencias opcionales
Si usa Axe en cuadernos Jupyter:

pip install git+https://github.com/facebook/Ax.git#egg=ax-platform[notebook]
Para admitir el trazado basado en diagramas en las versiones más recientes de Jupyter Notebook

pip install "notebook>=5.3" "ipywidgets==7.5"
Consulte el archivo README de Plotly repo para obtener detalles e instrucciones de JupyterLab.

Si almacena experimentos de Axe a través de SQLAlchemy en MySQL o SQLite:

pip install git+https://github.com/facebook/Ax.git#egg=ax-platform[mysql]
Únase a la comunidad Axe
Obteniendo ayuda
Abra un problema en nuestra página de problemas con preguntas, solicitudes de funciones o informes de errores. Si publica un informe de error, incluya un ejemplo reproducible mínimo (como un fragmento de código) que podamos usar para reproducir y depurar el problema que encontró.

contribuyendo
Vea el archivo CONTRIBUYENDO para saber cómo ayudar.

Al contribuir con Axe, recomendamos clonar el repositorio e instalar todas las dependencias opcionales:

# bleeding edge versions of GPyTorch + BoTorch are recommended
pip install git+https://github.com/cornellius-gp/gpytorch.git
pip install git+https://github.com/pytorch/botorch.git

git clone https://github.com/facebook/ax.git --depth 1
cd ax
pip install -e .[notebook,mysql,dev]
Consulte la recomendación anterior para instalar PyTorch para usuarios de MacOS.

El ejemplo anterior limita el tamaño del directorio clonado mediante el --depth argumento a git clone. Si necesita todo el historial de confirmaciones, puede eliminar este argumento.

Licencia
Axe está autorizado bajo la licencia MIT .
