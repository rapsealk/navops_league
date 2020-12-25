using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraController : MonoBehaviour
{
    public Warship TargetObject;

    // Start is called before the first frame update
    void Start()
    {
        //Cursor.visible = false;
        Cursor.lockState = CursorLockMode.Locked;
    }

    // Update is called once per frame
    void Update()
    {
        float horizontal = Input.GetAxis("Mouse X") * Time.deltaTime;
        float vertical = -1 * Input.GetAxis("Mouse Y") * Time.deltaTime;

        transform.RotateAround(TargetObject.transform.position, TargetObject.transform.up, horizontal * 100);
        transform.Rotate(Vector3.right, vertical * 50);

        Vector3 rotation = transform.rotation.eulerAngles;
        rotation.x = (rotation.x + 360) % 360;
        rotation.x = (rotation.x > 180f) ? (rotation.x - 360f) : rotation.x;
        if (Mathf.Abs(rotation.x) > 25f + Mathf.Epsilon)
        {
            rotation.x = Mathf.Sign(rotation.x) * 25f;
            transform.rotation = Quaternion.Euler(rotation);
        }

        //Debug.Log($"CameraController.rotation: {transform.rotation}");
        TargetObject.SetTargetPoint(transform.rotation);
    }
}
