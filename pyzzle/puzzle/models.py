from django.db import models

class PuzzleState(models.Model):
    state = models.CharField(max_length=100)
    parent = models.ForeignKey('self', on_delete=models.CASCADE, null=True)
    action = models.IntegerField(null=True)
    