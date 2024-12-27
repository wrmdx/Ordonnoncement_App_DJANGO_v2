from django.db import models

class Flow(models.Model):
    ch=[('SDST','SDST'),('no-idle','no-idle'),('no-wait','no-wait')]
    ch2=[('FIFO','FIFO'),('LIFO','LIFO'),('SPT','SPT'),('LPT','LPT')]
    nbrMachine=models.IntegerField()
    nbrJob=models.IntegerField()
    contrainte=models.CharField(max_length=150,choices=ch)
    critere=models.CharField(max_length=150,choices=ch2)

