A TENER EN CUENTA:

* SEMANA SANTA ES LA SEMANA 8, la he quitado
* APAGÓN LA SEMANA 28/04



Objetivos (Provisional):

Tratar de mejorar la estructura del grafo.

Seguir probando parámetros para modelCreator y modelTrainer.

Modularizar dataLoader, graphConstructor, ModelCreator, ModelTrainer de tal forma que, adaptando en un mínimo el dataLoader, reconozca cualquier estructura de archivos de datos de alumnos de una asignatura.

En vez de un modelo para toda asignatura, un modelo por asignatura que los profesores entrenen con datos de años anteriores y sea capaz de detectar alumnos en riesgo de suspenso a tiempo (como a mitad de cuatrimestre).

Buscar BDs más o menos completas para probar con otras asignaturas.

Tal vez tratar de definir una estructura común para cualquier archivo de datos que se le pase (Dividir características de estudiantes en resultados académicos, contexto personal, asistencia...) para poder crear grafos con alumnos de los que se aportan distintos datos. 



Resultados (Hasta el momento):

Predicción de la nota final en la BD provista, sin tomar como datos los resultados de los parciales, con un error de 0.8 puntos (sobre 10)



