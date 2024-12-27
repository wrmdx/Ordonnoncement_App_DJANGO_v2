from django.urls import path
from .views import (
    flowform2, showGantt, home, home2, GA_form,
    jobform1, jobform2, jobform3, showjobgantt, home_page
)

urlpatterns = [
    path('', home_page, name='home_page'),
    path('form', home2, name='home2'),
    path('flowshop', home, name='flowshop'),
    path('flowform2/<str:rows>/<str:columns>/<str:cont>', flowform2, name='flowform2'),
    path('gantt/<str:M>/<str:r>/<str:c>/<str:crit>/<str:cont>/<str:R>/<str:d>/<str:S>/<str:N>/<str:NG>/<str:Pm>',
         showGantt, name='showgantt'),
    path('GA/<str:M>/<str:r>/<str:c>/<str:crit>/<str:cont>/<str:R>/<str:d>/<str:S>',
         GA_form, name='GAform'),
    path('jobshop', jobform1, name='jobform1'),
    path('jobshop2/<str:J>/<str:m>', jobform2, name='jobform2'),
    path('jobshop3/<str:J>/<str:m>/<str:o>', jobform3, name='jobform3'),
    path('showJobshop/<str:J>/<str:m>/<str:o>/<str:O>/<str:p>/<str:crit>/<str:cont>',
         showjobgantt, name='showjobgantt'),
]