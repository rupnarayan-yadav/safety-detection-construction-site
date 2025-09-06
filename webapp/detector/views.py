

# from django.shortcuts import render

# # Create your views here.

# # alerts/views.py
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from .serializers import AlertSerializer

# class CreateAlertView(APIView):
#     def post(self, request, *args, **kwargs):
#         """
#         Receives alert data from the YOLO script and saves it.
#         """
#         serializer = AlertSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             print(f"✅ Alert Received: {serializer.data.get('violation_type')}")
#             # You can add email/notification logic here
#             return Response(serializer.data, status=status.HTTP_201_CREATED)

#         print(f"❌ Invalid data received: {serializer.errors}")
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)








from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import AlertSerializer

class CreateAlertView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = AlertSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            print(f"✅ Alert Received: {serializer.data.get('violation_type')}")
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        print(f"❌ Invalid data received: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)






