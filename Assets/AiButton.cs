using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class AiButton : MonoBehaviour
{
    public Button button;
    public GameManager gameManager;
    // Start is called before the first frame update
    void Start()
    {
        button = GetComponent<Button>();
        button.onClick.AddListener(delegate { toggleAI(); });
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void toggleAI()
    {
        for(int i=0;i< gameManager.m_Tanks.Length;i++)
        {
            gameManager.m_Tanks[i].m_Instance.GetComponent<TankMovement>().ToggleAi();
            gameManager.m_Tanks[i].m_Instance.GetComponent<TankShooting>().ToggleAi();
        }
    }
}
