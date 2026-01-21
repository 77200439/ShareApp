import streamlit as st
import pandas as pd
import pickle
import os
import shap
import matplotlib.pyplot as plt

# 加载模型
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 组合当前目录与模型文件名，生成模型的完整路径
model_path = os.path.join(current_dir, 'xgboost_model_260121.pkl')
# 打开并加载模型
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# 初始化SHAP解释器（XGBoost专用，放在模型加载后）
explainer = shap.TreeExplainer(model)  # 直接用原始XGB模型初始化
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

# 设置 Streamlit 应用的标题
st.title("Prediction of PCI based on Random Forest model")

st.sidebar.header("Selection Panel")
# st.sidebar.subheader("Picking up parameters")

# Smoke_option = ["Yes", "No"]
# Smoke_map = {"Yes": 1, "No": 0}
# Smoke_sb = st.sidebar.selectbox("Smoke", Smoke_option, index=0)

# SBP = st.sidebar.slider("SBP(mmHg)", min_value=88, max_value=225, value=90, step=1)

# PLT = st.sidebar.slider(r"PLT($10^3/uL$)", min_value=18, max_value=644, value=212, step=1)

# UA = st.sidebar.slider("UA(umol/L)", min_value=68, max_value=607, value=270, step=1)

# BAD = st.sidebar.slider("BAD(mm)", min_value=1.9, max_value=7.0, value=4.7, step=0.1)

# TG = st.sidebar.slider("TG(mg/dL)", min_value=0.46, max_value=7.32, value=1.73, step=0.01)

# BAL = st.sidebar.slider("RBC(mm)", min_value=0.2, max_value=36.4, value=27.0, step=0.1)

# BAMD = st.sidebar.slider("BAMD(mm)", min_value=0.0, max_value=12.5, value=0.0, step=0.1)

SBP = st.sidebar.slider(r"SBP(mmHg)", min_value=88, max_value=225, value=90, step=1)
PLT = st.sidebar.slider(r"PLT($10^3/\mu$L)", min_value=18, max_value=644, value=212, step=1)
UA = st.sidebar.slider(r"UA($\mu$mol/L)", min_value=68, max_value=607, value=270, step=1)
TG = st.sidebar.slider(r"TG(mg/dL)", min_value=0.46, max_value=7.32, value=1.73, step=0.01)
BAD = st.sidebar.slider(r"BAD(mm)", min_value=1.9, max_value=7.0, value=4.7, step=0.1)
BAL = st.sidebar.slider(r"BAL(mm)", min_value=0.2, max_value=36.4, value=27.0, step=0.1)
BAMD = st.sidebar.slider(r"BAMD(mm)", min_value=0.0, max_value=12.5, value=0.0, step=0.1)

input_data = pd.DataFrame({
    'SBP': [SBP],
    'PLT': [PLT],
    'UA': [UA],
    'TG': [TG],
    'BAD': [BAD],
    'BAL': [BAL],
    'BAMD': [BAMD]
})

if st.button("Calculate"):
    y_pred = model.predict(input_data)
    y_pred_proba = model.predict_proba(input_data)[:, 1]
    final_pred_proba = y_pred_proba[0] * 100
    # st.write(f"Predictive Probability: {final_pred_proba:.2f}%")
    st.markdown(f"### Predictive Probability: {final_pred_proba:.2f}%", unsafe_allow_html=True)

    shap_values = explainer.shap_values(input_data)
    shap_vals = shap_values[0] if len(shap_values.shape) == 2 else shap_values
    base_val = float(explainer.expected_value)

    shap_exp = shap.Explanation(
        values=shap_vals,
        base_values=base_val,
        data=input_data.iloc[0],
        feature_names=input_data.columns
    )
    # 绘图展示（去掉ax参数）
    st.subheader("SHAP Feature Contribution (Waterfall Plot)")
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_exp, show=False)
    plt.tight_layout()
    st.pyplot(plt.gcf())
