# from rest_framework import serializers
# from .models import Alert

# class AlertSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Alert
#         fields = ['id', 'timestamp', 'violation_type', 'camera_id', 'snapshot']







from rest_framework import serializers
from .models import Alert

class AlertSerializer(serializers.ModelSerializer):
    class Meta:
        model = Alert
        fields = [
            'id',
            'timestamp',
            'violation_type',
            'camera_id',
            'snapshot',
            'clip_path',
            'people_detected',
            'image_caption',
            'detected_text'
        ]



