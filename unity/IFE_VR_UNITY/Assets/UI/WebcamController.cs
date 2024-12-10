using UnityEngine;
using UnityEngine.UI;

public class OverlayController : MonoBehaviour
{
    public RectTransform canvas;
    public GameObject boxPrefab;

    public void DrawBoxes(string jsonData)
    {
        foreach (Transform child in canvas)
            Destroy(child.gameObject);

        var detections = JsonUtility.FromJson<Detection[]>(jsonData);
        foreach (var det in detections)
        {
            var box = Instantiate(boxPrefab, canvas);
            box.GetComponent<RectTransform>().anchoredPosition = new Vector2(det.box.x, det.box.y);
            box.GetComponent<RectTransform>().sizeDelta = new Vector2(det.box.width, det.box.height);
            box.GetComponentInChildren<Text>().text = det.label;
        }
    }
}

[System.Serializable]
public class Detection
{
    public string label;
    public Rect box;
}