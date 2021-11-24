options=$1

echo "Run trackEval with option: ${options}"

if [[ $options == "error_analysis" ]]
then
    echo "Evaluate and Error analysis"
    python scripts/run_mot_challenge.py --BENCHMARK MOT16 \
                                    --SPLIT_TO_EVAL test \
                                    --TRACKERS_TO_EVAL ch_yolov5m_deep_sort \
                                    --METRICS HOTA CLEAR Identity \
                                    --EXTRACTOR FP FN \
                                    --HEATMAP GT PRED IDSW \
                                    --ID_SWITCH True  
else
    echo "Evaluate only"
    python scripts/run_mot_challenge.py --BENCHMARK MOT16 \
                                    --SPLIT_TO_EVAL test \
                                    --TRACKERS_TO_EVAL ch_yolov5m_deep_sort \
                                    --METRICS HOTA CLEAR Identity
fi
