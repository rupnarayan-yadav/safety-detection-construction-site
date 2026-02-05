
from django.db import models

class Alert(models.Model):
    VIOLATION_CHOICES = [
        ('NO_HELMET', 'No Helmet Detected'),
        ('NO_MASK', 'No Mask Detected'),
        ('NO_SAFETY_VEST', 'No Safety Vest Detected'),
    ]

    timestamp = models.DateTimeField(auto_now_add=True)
    violation_type = models.CharField(max_length=50, choices=VIOLATION_CHOICES)
    camera_id = models.CharField(max_length=100, blank=True, null=True)
    snapshot = models.ImageField(upload_to='snapshots/', blank=True, null=True)

    # New fields for metadata
    clip_path = models.CharField(max_length=255, blank=True, null=True)
    people_detected = models.JSONField(blank=True, null=True)  # store list of names
    image_caption = models.TextField(blank=True, null=True)
    detected_text = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.get_violation_type_display()} at {self.timestamp.strftime('%Y-%m-%d %H:%M')}"


