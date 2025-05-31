### Order Rating Prediction Model Based on XGBoost-RandomForest Ensemble Learning

**Author**
Jianing Song
#### Abstract
This study focuses on the prediction of user ratings in e-commerce platforms, aiming to determine whether a customer is likely to leave a positive review based on structured features from historical transactions. To support service optimization and timely intervention for merchants, we utilize the publicly available e-commerce dataset from Brazil's Olist platform, which encompasses multi-dimensional information on orders, products, logistics, and customer feedback.

In the data processing stage, this paper performs binary classification labeling on the rating data. Through the **mutual information** method, it evaluates the feature importance of all variables and filters out nine key features most relevant to the rating results, including variables such as logistics timeliness, performance ratio, and order amount.

In terms of model construction, this paper attempts mainstream classifiers such as XGBoost and random forest, and finally fuses them into a **stacked ensemble model**, using logistic regression as a meta-learner to achieve non-linear integration of multi-model results. Additionally, to further alleviate the imbalance of rating labels, we introduce a **SMOTE oversampling** strategy to balance the training data.

In terms of model evaluation, we comprehensively used multiple classification metrics, as well as ROC curves, Precision-Recall curves, etc., for comparative analysis. The results show that the constructed Stacking ensemble model achieved the best performance on the test set, significantly outperforming single models.

The model not only maintains high overall accuracy but also significantly enhances the ability to identify negative reviews (minority class samples), demonstrating the effectiveness and robustness of ensemble learning approaches in the context of e-commerce review prediction tasks.

#### Rationale
In the era of prevalent digital shopping, if e-commerce platforms can intelligently predict order ratings based on multi-dimensional data such as user behavior, products, and logistics, they can achieve values including proactive intervention in negative review risks, optimization of product recommendations, assistance with merchant analysis, and enhancement of user satisfaction.

Compared with traditional rating analysis that relies on average scores or rules, rating prediction models built based on machine learning and big data mining integrate multi-dimensional information, offering stronger interpretability and generalizability. These models can effectively support the platform's personalized services and refined operations.

#### Research Question
Can we predict whether a customer will give a positive or negative review using only structured data-such as product information, logistics data, and order features-before the review is actually written, by leveraging machine learning models?

#### Data Sources
This project utilizes the publicly available e-commerce dataset from the Brazilian platform Olist, which captures the complete customer journey from registration, order placement, and product browsing to order fulfillment and review submission. Based on this dataset, a predictive model for user ratings is developed.

1. Dataset:
    There are 4 CSV files, containing 24 features such as time features, amount features, product features, and user behavior features, with a total of nearly 100,000 pieces of data

2. Target variable
    Positive evaluation (rating 4-5), negative evaluation (rating 1-3)

#### Methodolog
1. Data preprocessing
    This includes data cleaning and filtering for each file, feature engineering, descriptive statistical analysis, etc., with abnormal data removed and issues such as data imbalance identified.

2. Correlation analysis
    To identify the important variables affecting review ratings, we used the **Mutual Information (MI)** method to evaluate the non-linear dependence between each feature and the target variable.

    Finally, we selected the top 9 variables with the highest correlation, including delivery_time, shipping_time, and delivery_ratio, as the model inputs.

3.  Model Development

    - We compared the performance of four categories of mainstream machine learning models, namely Logistic Regression, Balanced Random Forest, XGBoost, and Stacking (an ensemble learning model using the first three as base models).

    - Through comparative analysis of multiple parameters, the Stacking ensemble model constructed with XGBoost and Random Forest as base models and Logistic Regression as the meta-learner was finally selected as the optimal prediction model for this task.

    - Determine the optimal parameter combination of the model through grid search.
 
    - Evaluate the performance of the base models through 5-fold cross-validation to prevent overfitting.

    - Use SMOTE oversampling to handle class imbalance.

4. Model Evaluation
    To comprehensively evaluate the performance of different models in the user rating prediction task, we conducted a comprehensive comparative analysis from three dimensions: classification metrics (F1, Precision, Recall), ROC curve (AUC), and Precision-Recall curve (AP).

#### Results
- **Model Performance**:  
  | Model                    | Accuracy | F1-Score | Recall |    AUC     |    AP     |
  |--------------------------|----------|----------|--------|------------|-----------|
  | Stacking                 | 80.2%    | 0.798    | 0.802  |  **0.79**  | **0.90**  |
  | XGBoost                  | 79.8%    | 0.698    | 0.783  |    0.50    |   0.74    |
  | RandomForest             | 82.2%    | 0.755    | 0.818  |    0.49    |   0.74    |
  | Logistic Regression      | 67.8%    | 0.616    | 0.692  |    0.61    |   0.80    |
  
- **Key Insights**:  
  - **Classification metrics** : The Stacking model achieves the optimal balance in F1, Recall, and Precision metrics, especially significantly leading in Recall-0 and F1-0, indicating its obvious advantages in minority class detection.

  - **ROC & PR Curve**: In terms of the ROC curve, the Stacking model achieved an AUC of 0.79, significantly outperforming other models. In the P-R curve evaluation, the Stacking model performed excellently again, with an average precision (AP) of 0.90, maintaining a high recall rate while ensuring prediction accuracy at high recall levels.

Based on the above evaluation results, the Stacking model demonstrates the best performance in terms of accuracy, robustness, and class balance handling. Therefore, this study finally selects the Stacking model as the rating prediction model.

#### Next steps
Based on the findings of this study, the following directions can be explored to further improve and expand the model's performance and application value in subsequent research:

- Introduce more dynamic behavioral features to enhance the model's ability to capture personalized rating tendencies.

- It is considered to introduce time-series features and special holiday markers (such as Christmas, etc.) to model the potential impact of temporal context on ratings

- It is considered to link the rating prediction task with other platform modules such as the recommendation system and after-sales system, so as to achieve prediction-driven refined operations and service optimization.

- Product reviews can be introduced, and large language models can be used to perform sentiment analysis on the text, which can further accurately predict product ratings.

#### Conclusion
This study constructs and validates the effectiveness of the StackingClassifier ensemble model in predicting user ratings in e-commerce scenarios, demonstrating significantly superior performance compared to traditional single models.

Despite these notable achievements, the model still has certain limitations. Its ability to identify negative ratings is relatively weak (with a recall rate of only 0.553), indicating room for improvement in recognizing dissatisfied users. Additionally, while SMOTE oversampling effectively mitigates class imbalance during training, actual deployment may be affected by data distribution drift or business changes, thereby reducing the model's generalization capability.

To enhance model robustness, future research could explore cost-sensitive learning, more advanced sampling techniques such as ADASYN, or attempt to incorporate textual information and multimodal approaches (e.g., Transformer architectures) to improve predictive power. At the deployment level, attention should also be paid to inference efficiency, latency control, and regular retraining of the model to adapt to dynamic changes in customer behavior. It is recommended to continuously monitor the model's fairness performance across different customer segments in practical applications to prevent overfitting to historical patterns.

In summary, while ensemble learning provides a robust baseline solution for review prediction tasks, ensuring its stable operation in production environments requires careful balancing between accuracy and sustainability.


### Bibliography 
[1] Y.-C. Huang and Y.-H. Lai, "Predicting learning achievement using ensemble learning with result explanation," PLOS ONE, vol. 19, no. 2, p. e0312124, 2024.

[2] L. Alzubaidi, J. F. Al-Shamma, M. A. Fadhel, O. Al-Khateeb, and M. A. M. Sabri, "A comprehensive review on ensemble deep learning: Opportunities and challenges," J. King Saud Univ. Comput. Inf. Sci., vol. 35, no. 4, pp. 344-363, 2023.

[3] J. R. Vergara and P. A. Estévez, "A review of feature selection methods based on mutual information,” Neural Comput. Appl., vol. 24, no. 1, pp. 175–186, 2014.

[4] T. T. Nguyen and H. T. Nguyen, “Predictive Model for Customer Satisfaction Analytics in E-commerce,” Journal of Retail Analytics, 2024. 

[5] S. Sharma, “Building a product rating predictor for fashion e-commerce using machine learning,” Medium, 2025.


##### Contact and Further Information
The Olist e-commerce dataset used in the project can be obtained through the following link：https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

For any questions, please contact: songjianing0206@hotmail.com