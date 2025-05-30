### Order Rating Prediction Model Based on XGBoost-RandomForest Ensemble Learning

**Author**
Jianing Song
#### Abstract
This study focuses on the prediction of user ratings in e-commerce platforms, aiming to determine whether a customer is likely to leave a positive review based on structured features from historical transactions. To support service optimization and timely intervention for merchants, we utilize the publicly available e-commerce dataset from Brazil's Olist platform, which encompasses multi-dimensional information on orders, products, logistics, and customer feedback.

During the data preprocessing stage, to address the significant class imbalance in the review labels (with positive reviews accounting for over 70%) and enhance prediction accuracy, the review scores were transformed into a binary classification task. Feature importance was evaluated using **mutual information**, based on which the top nine most relevant predictors were selected. These key features include logistics timeliness, fulfillment ratios, and total order value, all of which exhibit strong associations with the final review outcomes.

In terms of model construction, this study explored mainstream classifiers such as *XGBoost* and *Random Forest*, and ultimately combined them into a **stacking ensemble model**, with *logistic regression* employed as the meta-learner to achieve nonlinear integration of multiple model outputs. Furthermore, to mitigate the class imbalance in review labels, a **SMOTE oversampling** strategy was introduced to balance the training data, and feature standardization was applied during the preprocessing stage to enhance model stability.

In terms of model evaluation, we conducted a comparative analysis using multiple metrics, including **accuracy**, *precision*, *recall*, *F1-score*, *ROC curves*, and *precision-recall curves*. The results demonstrate that the constructed **stacking ensemble model** achieved the best performance on the test set, with an **accuracy** of 80.2%, a minority class F1-score of 0.588, and an average precision (AP) of 0.90 - significantly outperforming individual models such as *XGBoost* or *Logistic Regression*.

The model not only maintains high overall accuracy but also significantly enhances the ability to identify negative reviews (minority class samples), demonstrating the effectiveness and robustness of ensemble learning approaches in the context of e-commerce review prediction tasks.

#### Rationale
In today's increasingly digital shopping environment, users' experiences and satisfaction on e-commerce platforms are often reflected through rating systems. If the platform can intelligently predict user ratings for orders based on multi-dimensional data such as users' past behavior, product attributes, logistics information, and order and delivery times, it can generate value on multiple levels:

    - Proactive intervention for negative review risks: the system can trigger customer support or retention incentives in advance for orders predicted to receive low ratings.

    - Optimize product recommendation mechanisms: prioritize recommending products that users are more likely to rate highly.

    - Support merchant operational analysis: identify key factors influencing user ratings, such as slow logistics or discrepancies between product descriptions and actual items.

    - Enhance overall user satisfaction: achieve data-driven service improvement through a closed-loop system of prediction and optimization.

Traditional rating analysis often relies on average scores or rule-based heuristics, which tend to overlook individual differences in user rating behavior and contextual dependencies. By leveraging machine learning and big data mining techniques, we can integrate order details, user profiles, product attributes, and temporal features to build a rating prediction model with greater interpretability and generalizability. This provides robust support for personalized services and refined operations on e-commerce platforms.

#### Research Question
Can we predict whether a customer will give a positive or negative review using only structured data-such as product information, logistics data, and order features-before the review is actually written, by leveraging machine learning models?

#### Data Sources
This project utilizes the publicly available e-commerce dataset from the Brazilian platform Olist, which captures the complete customer journey from registration, order placement, and product browsing to order fulfillment and review submission. Based on this dataset, a predictive model for user ratings is developed.

1. Dataset:

    A total of four CSV files are used in this project, described as follows:
    - Order Items Table *(olist_order_items_dataset.csv)*: Lists the detailed items included in each order, including product ID, seller ID, price, and freight value. 
    - Orders Table *(olist_orders_dataset.csv)*: Describes the lifecycle of orders from September 2016 to October 2018, including order placement time, shipping time, delivery time, and order status.
    - Products Table *(olist_products_dataset.csv)*: Provides product attribute information such as category name, dimensions, and weight.
    - Review Information Table *(olist_order_reviews_dataset.csv)*: Contains user ratings for orders (from 1 to 5 stars).

2. Target variable
    Positive evaluation (rating 4-5), negative evaluation (rating 1-3)

#### Methodolog
1. Data preprocessing
    - Data Cleaning and Filtering
        Remove abnormal monthly data with extremely low order volumes; delete abnormal records with negative time differences; retain data where the order status is "delivered" and delete undelivered orders.

    - Missing Value Handling
        For categorical variables, missing values are filled with the mode, while for numerical variables (such as size and weight), missing values are filled with the mean.

    - Feature Engineering
     Time features: Extract date features such as year, month, and day of the week from order data.

     Order information features: Add three time difference features, including order approval duration, shipping duration, and delivery duration.

    - Descriptive Statistical Analysis
        Conduct descriptive statistics on the number of orders, the number of sellers, the number of products, prices, and freight costs.

        Statistically analyze the distribution of user ratings to identify data imbalance issues.

2. Correlation analysis
    To identify the important variables affecting review ratings, we used the **Mutual Information (MI)** method to evaluate the non-linear dependence between each feature and the target variable.

    Among them, delivery_time has the highest score in the normalized mutual information. Variables directly related to delivery efficiency and performance ratio, such as shipping_time and delivery_ratio, also show high correlation. Variables reflecting price structure and cost-performance ratio, such as freight_ratio and price_per_weight, also demonstrate certain information value, indicating the impact of freight burden and product pricing on customer perception.

    In contrast, variables related to product description dimensions such as product_photos_qty, product_height_cm, product_width_cm, and product_name_length have mutual information close to zero, indicating weak statistical correlation with rating outcomes. These fields are considered for exclusion.

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

  - **ROC & PR Curve**: In terms of the ROC curve, the AUC of the Stacking model is as high as 0.79, significantly outperforming other models. In the P-R curve evaluation, the Stacking model excels again, with an average precision (AP) of 0.90. It not only maintains a high recall rate but also ensures prediction accuracy at high recall levels, reflecting its stronger ability to identify negative reviews and lower misjudgment rate in practical business scenarios.

Based on the above evaluation results, the Stacking model demonstrates the best performance in terms of accuracy, robustness, and class balance handling. It effectively integrates the advantages of multiple base models (XGBoost, RandomForest, Logistic Regression) and can stably capture complex features and boundary samples in rating tendencies. Therefore, this study finally selects the Stacking model as the rating prediction model.

#### Next steps
Based on the findings of this study, the following directions can be explored to further improve and expand the model's performance and application value in subsequent research:

- Incorporate more dynamic behavioral features, such as users' historical rating preferences and other user behavior data, to enhance the model's ability to capture personalized rating tendencies

- It is considered to introduce time-series features and special holiday markers (such as Christmas, etc.) to model the potential impact of temporal context on ratings

- It is considered to link the rating prediction task with other platform modules (such as recommendation systems and after-sales systems) to achieve prediction-driven refined operations and service optimization, ultimately promoting the improvement of overall service quality and user stickiness.

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