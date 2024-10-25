from django.db import models

class Career(models.Model):
    
    id_career = models.AutoField(primary_key=True)
    career = models.CharField(max_length=30)

    def __str__(self):
        return self.career

    class Meta:
        managed = False  
        db_table = 'career'  


class Semester(models.Model):
    id_semester = models.AutoField(primary_key=True)
    semester = models.CharField(max_length=30)

    def __str__(self):
        return self.semester

    class Meta:
        managed = False
        db_table = 'semester'

class Student(models.Model):
    id_student = models.AutoField(primary_key=True)
    name_student = models.CharField(max_length=30)
    registration_student = models.CharField(max_length=30)
    semester = models.ForeignKey(Semester, on_delete=models.CASCADE, db_column='id_semester') 
    career = models.ForeignKey(Career, on_delete=models.CASCADE, db_column='id_career') 

    def __str__(self):
        return self.name_student

    class Meta:
        managed = False
        db_table = 'students'

class Register(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    emotion_type = models.CharField(max_length=20, blank=True, null=True)
    accuracy = models.FloatField()

    def __str__(self):
        return f"Register {self.id} for {self.student.name_student}"

    class Meta:
        managed = False
        db_table = 'registers'
