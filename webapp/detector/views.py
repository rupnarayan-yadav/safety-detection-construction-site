# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from .serializers import AlertSerializer

# class CreateAlertView(APIView):
#     def post(self, request, *args, **kwargs):
#         serializer = AlertSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             print(f"✅ Alert Received: {serializer.data.get('violation_type')}")
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         print(f"❌ Invalid data received: {serializer.errors}")
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)





# detector/views.py

from django.shortcuts import render, get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import AlertSerializer
from .models import Alert

# --- API View (You already have this) ---
class CreateAlertView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = AlertSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# --- Frontend Views (Add these) ---
def dashboard(request):
    """
    Renders the main dashboard page. The alert list will be loaded dynamically.
    """
    return render(request, 'detector/dashboard.html')

def alert_list_partial(request):
    """
    Fetches all alerts and renders only the list portion of the HTML.
    This is used for AJAX updates on the dashboard.
    """
    alerts = Alert.objects.all().order_by('-timestamp')
    return render(request, 'detector/partials/alert_list.html', {'alerts': alerts})

def alert_detail(request, pk):
    """
    Displays the full details for a single alert.
    """
    alert = get_object_or_404(Alert, pk=pk)
    return render(request, 'detector/alert_detail.html', {'alert': alert})