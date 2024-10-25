CREATE TABLE students(
	id_student SERIAL PRIMARY KEY,
	name_student VARCHAR(30) NOT NULL,
	registration_student VARCHAR(30) NOT NULL,
	id_semester INT NOT NULL,
	id_career INT NOT NULL,
	FOREIGN KEY (id_semester) REFERENCES semester(id_semester),
	FOREIGN KEY (id_career) REFERENCES career(id_career)
);


CREATE TABLE career(
	id_career SERIAL PRIMARY KEY,
	career VARCHAR(30) NOT NULL
);

CREATE TABLE semester(
	id_semester SERIAL PRIMARY KEY,
	semester VARCHAR(30) NOT NULL

);


CREATE TABLE registers (
	
	id_register SERIAL PRIMARY KEY,
	id_student INT NOT NULL,
	emotion_type VARCHAR(20),
	accuracy FLOAT NOT NULL,
	FOREIGN KEY (id_student) REFERENCES students(id_student)

);


INSERT INTO career (career)
VALUES 	( 'ING SISTEMAS COMPUTACIONALES'),
		( 'ING AMBIENTAL'),
		( 'ING ELECTRONICA'),
		( 'ING BIOMEDICA'),
		( 'ING INFORMATICA'),
		( 'LICENCIATURA EN ADMINISTRACION'),
		( 'ARQUITECTURA');


INSERT INTO semester (semester)
VALUES	( 'PRIMERO'),
		( 'SEGUNDO'),
		( 'TERCERO'),
		( 'CUARTO'),
		( 'QUINTO'),
		( 'SEXTO'),
		( 'SEPTIMO'),
		( 'OCTAVO'),
		( 'NOVENO'),
		( 'DECIMO');