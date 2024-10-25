from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from rest_framework import status
    
from .serializer import CareerSerializer
from .models import Career

@require_http_methods(["GET"])
def get_careers(request):
    
    careers = Career.objects.all()
    careersSerializer = CareerSerializer(careers, many=True)

    return JsonResponse({'status': status.HTTP_200_OK, 'data' : careersSerializer.data})
  
