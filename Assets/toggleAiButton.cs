using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class toggleAiButton : MonoBehaviour
{
    // Start is called before the first frame update
    public Toggle myToggle;
    public void ValueChanged()
    {
        Debug.Log("Value Changed");
    }

    public void ChangeToggle()
    {
        myToggle.isOn = !myToggle.isOn;
    }
}
