from django import forms
from .models import Flow

class flow_form(forms.ModelForm):
    class Meta:
        model=Flow
        fields=['nbrMachine','nbrJob','contrainte','critere']


