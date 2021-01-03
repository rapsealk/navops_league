using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Radar : MonoBehaviour
{
    public Transform Player;
    public Transform Opponent;
    public RectTransform UserInterface;
    public GameObject BlipRedPrefab;
    //public GameObject BlipBluePrefab;
    public float RadarRange = 20f;
    //public float BlipSize = 15f;
    public float ViewDirection = 0f;

    private float radarWidth;
    private float radarHeight;
    private float blipWidth;
    private float blipHeight;

    // Start is called before the first frame update
    void Start()
    {
        radarWidth = UserInterface.rect.width;
        radarHeight = UserInterface.rect.height;
        blipWidth = radarWidth / 10;
        blipHeight = radarHeight / 10;
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 rotation = transform.rotation.eulerAngles;
        rotation.y += 360f * Time.deltaTime;
        transform.rotation = Quaternion.Euler(rotation);

        // Update User Interface
        GameObject[] blips = GameObject.FindGameObjectsWithTag("RadarBlip");
        for (int i = 0; i < blips.Length; i++)
        {
            Destroy(blips[i]);
        }

        Vector3 position = Player.transform.position;
        Vector3 targetPosition = Opponent.transform.position;
        float distance = Vector3.Distance(position, targetPosition);
        Debug.Log($"Radar(target: {targetPosition}, distance: {distance}/{RadarRange})");
        if (distance <= RadarRange)
        {
            Vector3 normalizedTargetPosition = (targetPosition - position) / RadarRange;
            normalizedTargetPosition.y = 0f;
            targetPosition = normalizedTargetPosition;

            float angleToTarget = Mathf.Atan2(targetPosition.x, targetPosition.z) * Mathf.Rad2Deg;
            //float anglePlayer = Player.transform.rotation.eulerAngles.y;   // 0f;
            float anglePlayer = ViewDirection;
            float angleRadarDegrees = angleToTarget - anglePlayer - 90f;
            float normalizedDistanceToTarget = targetPosition.magnitude;
            float angleRadians = angleRadarDegrees * Mathf.Deg2Rad;
            float x = normalizedDistanceToTarget * Mathf.Cos(angleRadians) * radarWidth * 0.5f;
            float y = normalizedDistanceToTarget * Mathf.Sin(angleRadians) * radarHeight * 0.5f;
            x += (radarWidth * 0.5f) - blipWidth * 0.5f;
            y += (radarHeight * 0.5f) - blipHeight * 0.5f;
            Vector2 blipPosition = new Vector2(x, y);
            Debug.Log($"BlipPosition: {blipPosition}");

            GameObject blip = Instantiate(BlipRedPrefab);
            blip.transform.SetParent(UserInterface.transform);
            RectTransform rt = blip.GetComponent<RectTransform>();
            rt.SetInsetAndSizeFromParentEdge(RectTransform.Edge.Left, blipPosition.x, blipWidth);
            rt.SetInsetAndSizeFromParentEdge(RectTransform.Edge.Top, blipPosition.y, blipHeight);
        }
    }
}
