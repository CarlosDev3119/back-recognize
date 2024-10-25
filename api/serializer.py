from rest_framework import serializers
from .models import Career, Semester, Student

class CareerSerializer(serializers.ModelSerializer):
    class Meta: 
        model = Career
        fields = '__all__'
        

class SemesterSerializer(serializers.ModelSerializer):
    class Meta: 
        model = Semester
        fields = '__all__'
        
class StudentSerializer(serializers.ModelSerializer):
    class Meta: 
        model = Student
        fields = '__all__'