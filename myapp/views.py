import joblib
from django.shortcuts import render
import numpy
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime, timedelta


# Create your views here.


# model = load_model("model2.h5")
model = load_model("lstm2_model.h5")
scaler = joblib.load("scaler.pkl")
test_data = joblib.load("sc_test_data.pkl")

model.make_predict_function()



def myform(request):
    if request.method == "POST":
        hour = request.POST.get('hour')
        print("Hour : ",hour)

        # x_input = test_data[9961:].reshape(1, -1)
        x_input = test_data[9097:].reshape(1, -1)

        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        date_time_str = '31/12/18 23:50:00'

        date_time_obj = datetime.strptime(date_time_str, '%d/%m/%y %H:%M:%S')

        new_dates = []

        for i in range ((int(hour)*6)):
            date_time_obj += timedelta(minutes=10)
            new_dates.append(date_time_obj)


        print(new_dates)

        lst_output = []
        n_steps = 1008
        i = 0
        while (i < (int(hour)*6)):

            if (len(temp_input) > 1008):
                # print(temp_input)
                x_input = np.array(temp_input[1:])
                # print("{} day input {}".format(i, x_input))
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                # print(x_input)
                yhat = model.predict(x_input, verbose=0)
                # print("{} day output {}".format(i, yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                # print(temp_input)
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i = i + 1

        print(lst_output)
        pred_data = scaler.inverse_transform(lst_output)
        print(pred_data)

        result = []
        for i in range(len(new_dates)):
            result.append([str(new_dates[i]),  float("{:.2f}".format(pred_data[i][0])) ])

        return render(request, 'myapp/myform.html',{'result':result})
    else:

        return render(request, 'myapp/myform.html')