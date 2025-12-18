import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from pandas.io.formats.style import Styler
from typing import Optional, Union
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def classification_report_as_df(
        y_true: ArrayLike, 
        y_pred: ArrayLike,
        decimal_places_for_display: Optional[int] = None
    )->Union[pd.DataFrame, Styler]: 
    """Return a scikit-learn-style classification report as a pandas DataFrame.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values vector.
    y_pred : array-like
        Predicted target values vector.
    decimal_places_for_display : Optional[int], optional
        Number of decimal places to format the floating point values for display.
        If None (default), no specific formatting is applied. 

    Returns
    -------
    pandas.DataFrame or pandas.io.formats.style.Styler
        DataFrame representing the string-based classification report from
        scikit-learn, or a styled DataFrame if ``decimal_places_for_display`` is
        specified.

    Notes
    -----
    - Zero-division cases yield 0 for precision/recall/f1.
    """
    PRECISION_COL = 'precision'
    RECALL_COL = 'recall'
    F1_SCORE_COL = 'f1-score'
    SUPPORT_COL = 'support'

    MICRO = 'micro'
    MACRO = 'macro'
    ACCURACY = 'accuracy'

    accuracy = accuracy_score(y_true,y_pred)     
    accuracy_support = len(y_pred)      
    metrics_per_classes = np.array(list(precision_recall_fscore_support(y_true, y_pred,zero_division=0))).T      
    micro_metrics = list(     
        precision_recall_fscore_support(         
            y_true, y_pred, average=MICRO, zero_division=0     
            )
        )     
    micro_metrics[-1]=accuracy_support      
    macro_metrics = list(precision_recall_fscore_support(         
        y_true, y_pred, average=MACRO, zero_division=0     
        )
    )     
    macro_metrics[-1]=accuracy_support      
    report_index = list([str(k) for k in range(metrics_per_classes.shape[0])])     
    report_columns = [PRECISION_COL, RECALL_COL, F1_SCORE_COL, SUPPORT_COL]     
    report_frame = pd.DataFrame(np.nan, index=report_index, columns=report_columns)     
    report_frame.loc[report_index]=metrics_per_classes      
    accuracy_row = pd.DataFrame(         
        {             
            PRECISION_COL: '',              
            RECALL_COL: '',             
            F1_SCORE_COL:accuracy,             
            SUPPORT_COL:accuracy_support         
        },          
        index=[ACCURACY]     
    )     
    micro_row = pd.DataFrame(         
        {             
            PRECISION_COL: micro_metrics[0],              
            RECALL_COL: micro_metrics[1],             
            F1_SCORE_COL: micro_metrics[2],             
            SUPPORT_COL:micro_metrics[3]         
        },          
        index=[MICRO]     
    )     
    macro_row = pd.DataFrame(         
        {             
            PRECISION_COL: macro_metrics[0],              
            RECALL_COL: macro_metrics[1],             
            F1_SCORE_COL: macro_metrics[2],             
            SUPPORT_COL: macro_metrics[3]         
        },          
        index=[MACRO]     
    )       
    report_frame = pd.concat([report_frame,accuracy_row,micro_row,macro_row])     
      
    report_frame[report_frame.columns[-1]]=report_frame[report_frame.columns[-1]].astype(int)  

    if decimal_places_for_display is None:    
        return report_frame
    else:
        return report_frame.style.format(precision=decimal_places_for_display)
