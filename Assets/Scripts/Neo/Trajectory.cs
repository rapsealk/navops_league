using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Trajectory : MonoBehaviour
{
    [SerializeField]
    public Transform TargetObject;
    [Range(1.0f, 6.0f)]
    public float TargetRadius;
    [Range(20.0f, 75.0f)]
    public float LaunchAngle;
    [Range(0.0f, 10.0f)]
    public float TargetHeightOffsetFromGround;
    public bool RandomizeHeightOffset;

    public Vector3 InitialPosition { get; private set; }
    public Quaternion InitialRotation { get; private set; }

    private Rigidbody Rigidbody;
    private bool bTargetReady;
    // Start is called before the first frame update
    void Start()
    {
        Rigidbody = GetComponent<Rigidbody>();
        bTargetReady = true;
        InitialPosition = transform.position;
        InitialRotation = transform.rotation;
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (bTargetReady)
            {
                Launch();
            }
            else
            {
                ResetToInitialState();
                SetNewTarget();
            }
        }

        if (Input.GetKeyDown(KeyCode.R))
        {
            ResetToInitialState();
        }
    }

    private void Launch()
    {
        Vector3 projectileXZPos = new Vector3(transform.position.x, 0f, transform.position.z);
        Vector3 targetXZPos = new Vector3(TargetObject.transform.position.x, 0f, TargetObject.transform.position.z);

        transform.LookAt(targetXZPos);

        float R = Vector3.Distance(projectileXZPos, targetXZPos);
        float G = Physics.gravity.y;
        float tanAlpha = Mathf.Tan(LaunchAngle * Mathf.Deg2Rad);
        float H = (TargetObject.transform.position.y + GetPlatformOffset()) - transform.position.y;

        float Vz = Mathf.Sqrt(G * R * R / (2f * (H - R * tanAlpha)));
        float Vy = tanAlpha * Vz;

        Vector3 localVelocity = new Vector3(0f, Vy, Vz);
        Vector3 globalVelocity = transform.TransformDirection(localVelocity);
        
        Rigidbody.velocity = globalVelocity;
        bTargetReady = false;
    }

    private float GetPlatformOffset()
    {
        return 0f;
    }

    private void SetNewTarget()
    {
        Transform targetTransform = TargetObject.GetComponent<Transform>();
        Vector3 rotationAxis = Vector3.up;

        float randomAngle = Random.Range(0f, 360f);
        Vector3 randomVectorOnGroupPlane = Quaternion.AngleAxis(randomAngle, rotationAxis) * Vector3.right;
        float heightOffset = (RandomizeHeightOffset ? Random.Range(0.2f, 1.0f) : 1.0f) * TargetHeightOffsetFromGround;
        float aboveOrBlowGround = 1f;//Random.Range(0.0f, 1.0f) > 0.5f ? 1.0f : -1.0f;
        Vector3 heightOffsetVector = new Vector3(0f, heightOffset, 0f) * aboveOrBlowGround;
        Vector3 randomPoint = randomVectorOnGroupPlane * TargetRadius + heightOffsetVector;

        TargetObject.SetPositionAndRotation(randomPoint, targetTransform.rotation);
        bTargetReady = true;
    }

    private void ResetToInitialState()
    {
        Rigidbody.velocity = Vector3.zero;
        transform.SetPositionAndRotation(InitialPosition, InitialRotation);
        bTargetReady = false;
    }
}
