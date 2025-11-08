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
model_path = os.path.join(current_dir, 'xgboost_model_251108.pkl')
# 打开并加载模型
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# 初始化SHAP解释器（XGBoost专用，放在模型加载后）
explainer = shap.TreeExplainer(model)  # 直接用原始XGB模型初始化
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

# 设置 Streamlit 应用的标题
st.title("Prediction of migraine in patients with patent foramen ovale based on XGBoost model")

st.sidebar.header("Selection Panel")
# st.sidebar.subheader("Picking up parameters")

Sex_option = ["Male", "Female"]
Sex_map = {"Male": 1, "Female": 0}
Sex_sb = st.sidebar.selectbox("Sex", Sex_option, index=1)

Smoking_option = ["Yes", "No"]
Smoking_map = {"Yes": 1, "No": 0}
Smoking_sb = st.sidebar.selectbox("Smoking", Smoking_option, index=0)

Hypertension_option = ["Yes", "No"]
Hypertension_map = {"Yes": 1, "No": 0}
Hypertension_sb = st.sidebar.selectbox("Hypertension", Hypertension_option, index=1)

Atrial_fibrillation_option = ["Yes", "No"]
Atrial_fibrillation_map = {"Yes": 1, "No": 0}
Atrial_fibrillation_sb = st.sidebar.selectbox("Atrial_fibrillation", Atrial_fibrillation_option, index=1)

Left_atrial_shunt_option = ["Yes", "No"]
Left_atrial_shunt_map = {"Yes": 1, "No": 0}
Left_atrial_shunt_sb = st.sidebar.selectbox("Left_atrial_shunt", Left_atrial_shunt_option, index=0)

Age = st.sidebar.slider("Age, year", min_value=18, max_value=82, value=27, step=1)
RV = st.sidebar.slider("RV, mm", min_value=19, max_value=40, value=35, step=1)
RA = st.sidebar.slider("RA, mm", min_value=22, max_value=40, value=36, step=1)

input_data = pd.DataFrame({
    'Age': [Age],
    'Sex': [Sex_map[Sex_sb]],
    'Smoking': [Smoking_map[Smoking_sb]],
    'Hypertension': [Hypertension_map[Hypertension_sb]],
    'Atrial_fibrillation': [Atrial_fibrillation_map[Atrial_fibrillation_sb]],
    'Left_atrial_shunt': [Left_atrial_shunt_map[Left_atrial_shunt_sb]],
    'RV': [RV],
    'RA': [RA]
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
