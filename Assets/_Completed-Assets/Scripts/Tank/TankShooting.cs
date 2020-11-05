using System;
using Unity.UNetWeaver;
using UnityEngine;
using UnityEngine.UI;

namespace Complete
{
    public class TankShooting : MonoBehaviour
    {

        public bool m_UsingAI = true;
        public int m_PlayerNumber = 1;              // Used to identify the different players.
        public Rigidbody m_Shell;                   // Prefab of the shell.
        public Transform m_FireTransform;           // A child of the tank where the shells are spawned.
        public Slider m_AimSlider;                  // A child of the tank that displays the current launch force.
        public AudioSource m_ShootingAudio;         // Reference to the audio source used to play the shooting audio. NB: different to the movement audio source.
        public AudioClip m_ChargingClip;            // Audio that plays when each shot is charging up.
        public AudioClip m_FireClip;                // Audio that plays when each shot is fired.
        public float m_MinLaunchForce = 15f;        // The force given to the shell if the fire button is not held.
        public float m_MaxLaunchForce = 30f;        // The force given to the shell if the fire button is held for the max charge time.
        public float m_MaxChargeTime = 0.75f;       // How long the shell can charge for before it is fired at max force.


        private string m_FireButton;                // The input axis that is used for launching shells.
        private float m_CurrentLaunchForce;         // The force that will be given to the shell when the fire button is released.
        private float m_ChargeSpeed;                // How fast the launch force increases, based on the max charge time.
        private bool m_Fired;                       // Whether or not the shell has been launched with this button press.

        private string m_PitchCanonRotationButton;
        private string m_YawCanonRotationButton;
        public float m_PitchCanonRotationSpeed;
        public float m_YawCanonRotationSpeed;
        public GameObject m_TankTurret;
        public GameObject m_Target = null;

        public GameObject m_GameManager;

        //Shot calculation
        private float m_TargetDistance;
        public float m_ShotSpeed;
        public float m_ShotTimer = 3.0f;


        private void OnEnable()
        {
            // When the tank is turned on, reset the launch force and the UI
            m_CurrentLaunchForce = m_MinLaunchForce;
            m_AimSlider.value = m_MinLaunchForce;
        }

        private void Start ()
        {
            // The fire axis is based on the player number.
            m_FireButton = "Fire" + m_PlayerNumber;
            m_PitchCanonRotationButton = "PitchCanonRotation" + m_PlayerNumber;
            m_YawCanonRotationButton = "YawCanonRotation" + m_PlayerNumber;

            // The rate that the launch force charges up is the range of possible forces by the max charge time.
            m_ChargeSpeed = (m_MaxLaunchForce - m_MinLaunchForce) / m_MaxChargeTime;
        }


        private void Update ()
        {
            if (m_UsingAI)
            {
                m_ShotTimer -= Time.deltaTime;
                RotateCanon();

                if (m_ShotTimer <= 0.0f)
                {
                    Fire();
                    m_ShotTimer = 3.0f;
                }
            }
            else
            {
                RotateCanon();
                // The slider should have a default value of the minimum launch force.
                m_AimSlider.value = m_MinLaunchForce;

                // If the max force has been exceeded and the shell hasn't yet been launched...
                if (m_CurrentLaunchForce >= m_MaxLaunchForce && !m_Fired)
                {
                    // ... use the max force and launch the shell.
                    m_CurrentLaunchForce = m_MaxLaunchForce;
                    Fire();
                }
                // Otherwise, if the fire button has just started being pressed...
                else if (Input.GetButtonDown(m_FireButton))
                {
                    // ... reset the fired flag and reset the launch force.
                    m_Fired = false;
                    m_CurrentLaunchForce = m_MinLaunchForce;

                    // Change the clip to the charging clip and start it playing.
                    m_ShootingAudio.clip = m_ChargingClip;
                    m_ShootingAudio.Play();
                }
                // Otherwise, if the fire button is being held and the shell hasn't been launched yet...
                else if (Input.GetButton(m_FireButton) && !m_Fired)
                {
                    // Increment the launch force and update the slider.
                    m_CurrentLaunchForce += m_ChargeSpeed * Time.deltaTime;

                    m_AimSlider.value = m_CurrentLaunchForce;
                }
                // Otherwise, if the fire button is released and the shell hasn't been launched yet...
                else if (Input.GetButtonUp(m_FireButton) && !m_Fired)
                {
                    // ... launch the shell.
                    Fire();
                }
            }
        }

        private void RotateCanon()
        {
            if (m_UsingAI)
            {

                var q = Quaternion.LookRotation(m_Target.transform.position - transform.position);
                m_TankTurret.transform.rotation = Quaternion.RotateTowards(m_TankTurret.transform.rotation, q, 100 * Time.deltaTime);
            }
            else
            {
                float pitchRotation = Input.GetAxis(m_PitchCanonRotationButton) * m_PitchCanonRotationSpeed * Time.deltaTime;
                m_TankTurret.transform.Rotate(new Vector3(pitchRotation, 0f, 0f));

                float yawRotation = Input.GetAxis(m_YawCanonRotationButton) * m_YawCanonRotationSpeed * Time.deltaTime;
                m_TankTurret.transform.Rotate(new Vector3(0f, yawRotation, 0f));
            }
        }

        private void Fire ()
        {
            if (m_UsingAI)
            {
                m_TargetDistance = Vector3.Distance(m_FireTransform.position, m_Target.transform.position);

                float calc = Physics.gravity.y * (m_TargetDistance * m_TargetDistance);//; +2 * 0
                double calc1 = Math.Sqrt( (m_ShotSpeed * m_ShotSpeed * m_ShotSpeed * m_ShotSpeed) - Physics.gravity.y * (calc));
                double tangent = ( (m_ShotSpeed * m_ShotSpeed) - calc1 ) 
                    / (Physics.gravity.y * m_TargetDistance);

                double Rad = Math.Atan(tangent);

                if(Math.Abs((float)Rad * Mathf.Rad2Deg) > 45 || float.IsNaN(Math.Abs((float)Rad * Mathf.Rad2Deg)))
                {
                    Debug.Log("Can't fire, too far");
                    return; 
                }

                Debug.Log((float)Rad * Mathf.Rad2Deg);

                m_TankTurret.transform.Rotate((float)Rad * Mathf.Rad2Deg, 0.0f, 0.0f );
            }

            // Set the fired flag so only Fire is only called once.

            m_Fired = true;
            // Create an instance of the shell and store a reference to it's rigidbody.
            Rigidbody shellInstance =
                Instantiate(m_Shell, m_FireTransform.position, m_FireTransform.rotation) as Rigidbody;

            if (m_UsingAI)
            {
                shellInstance.velocity = m_ShotSpeed * m_FireTransform.forward;
            }
            else
            {
                // Set the shell's velocity to the launch force in the fire position's forward direction.
                shellInstance.velocity = m_CurrentLaunchForce * m_FireTransform.forward;
            }

            // Change the clip to the firing clip and play it.
            m_ShootingAudio.clip = m_FireClip;
            m_ShootingAudio.Play();

            // Reset the launch force.  This is a precaution in case of missing button events.
            m_CurrentLaunchForce = m_MinLaunchForce;
            
        }
    }
}