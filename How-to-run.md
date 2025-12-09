# How to run 

## 1. Create venv and install requirements

### **Windows**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### **Linux**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Run 

### **fuel_burn_api**
```bash
python fuel_burn_api.py
```

### **fuel_burn_streamlit**
```bash
streamlit run fuel_burn_streamlit.py 
```

### **vm_api**
```bash
python vm_api.py
```

### **vm_streamlit**
```bash
streamlit run vm_streamlit.py --server.port 8502
```