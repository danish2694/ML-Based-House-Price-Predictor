from django.shortcuts import render
from django.http import HttpResponse
from HPP import House_Price_Model

# Create your views here.

def index(request):
	return render (request,'index.html')

def prediction(request):
	if request.method == 'POST':
		total_sqft = float(request.POST.get('area'))
		bhk = int(request.POST.get('bhk'))
		bath = int(request.POST.get('bath'))
		location = request.POST.get('loc')
	params = House_Price_Model.predict_price(location,total_sqft,bhk,bath)
	print(params)
	return render (request,'index.html',{'data':params,'unit':'lakh'})