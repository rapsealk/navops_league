using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraController : MonoBehaviour
{
    public Transform TargetObject;

    // Start is called before the first frame update
    void Start()
    {
        Cursor.visible = false;
        Cursor.lockState = CursorLockMode.Locked;
    }

    // Update is called once per frame
    void Update()
    {
        float horizontal = Input.GetAxis("Mouse X") * Time.deltaTime;
        float vertical = -1 * Input.GetAxis("Mouse Y") * Time.deltaTime;

        transform.RotateAround(TargetObject.transform.position, TargetObject.transform.up, horizontal * 100);
        transform.Rotate(Vector3.right, vertical * 50);
    }
}
