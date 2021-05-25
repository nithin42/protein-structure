from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

clf=pickle.load(open('clf.sav','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
def create_dict(codes):
  char_dict = {}
  for index, val in enumerate(codes):
    char_dict[val] = index+1

  return char_dict

char_dict = create_dict(codes)




@app.route('/detect',methods=['POST','GET'])

def predict():
    
    int_features=request.form["u_data"]
    dat = int_features
#    testing = integer_encoding(int_features)

#def integer_encoding(testing):
    encode_list = []
    for row in dat:
        row_encode = []
        for code in row:
          row_encode.append(char_dict.get(code, 0))
        encode_list.append(np.array(row_encode))
#    return encode_list
    

    test_2 = encode_list
    #temp = temp.reshape(-1,1)
    newArray = np.ravel(test_2)
    newArray
    print(newArray)
    
    prediction=clf.predict([newArray[:100]])
    
   
    if prediction == 1:
        return render_template('index.html',pred='Endolysins')
    if prediction == 0:
        return render_template('index.html',red='Antolysins')



if __name__ == '__main__':
    app.run(debug=True)
    app.run(host="0.0.0.0", port="33")

