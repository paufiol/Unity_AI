using Pada1.BBCore;
using Pada1.BBCore.Tasks;
using UnityEngine;

namespace BBUnity.Actions
{
    /// <summary>
    /// It is an action to move the GameObject to a given position.
    /// </summary>
    [Action("Navigation/Wander")]
    [Help("make an object wander")]
    public class Wander : GOAction
    {

        private UnityEngine.AI.NavMeshAgent m_Agent;
        private Rigidbody m_rigidBody;
        private float radius = 1.0f;
        private float offset = 4.0f;
        private bool start;

        /// <summary>Initialization Method of MoveToPosition.</summary>
        /// <remarks>Check if there is a NavMeshAgent to assign a default one and assign the destination to the NavMeshAgent the given position.</remarks>
        public override void OnStart()
        {
            start = true;
            if (start)
            {
                m_Agent = gameObject.GetComponent<UnityEngine.AI.NavMeshAgent>();
                if (m_Agent == null)
                {
                    Debug.LogWarning("The " + gameObject.name + " game object does not have a Nav Mesh Agent component to navigate. One with default values has been added", gameObject);
                    m_Agent = gameObject.AddComponent<UnityEngine.AI.NavMeshAgent>();
                }

                Vector3 localTarget = new Vector3(Random.Range(-1.0f, 1.0f), 0, Random.Range(-1.0f, 1.0f));
                localTarget.Normalize();
                localTarget *= radius;

                Vector3 direction = gameObject.transform.forward;
                direction.Normalize();
                direction *= offset;

                localTarget += new Vector3(0.0f, 0.0f, offset);

                Vector3 worldTarget = gameObject.transform.TransformPoint(localTarget);
                worldTarget.y = 0.0f;

                m_Agent.destination = worldTarget;
            }

#if UNITY_5_6_OR_NEWER
            m_Agent.isStopped = false;
#else
                navAgent.Resume();
#endif
        }

        /// <summary>Method of Update of MoveToPosition </summary>
        /// <remarks>Check the status of the task, if it has traveled the road or is close to the goal it is completed
        /// and otherwise it will remain in operation.</remarks>
        public override TaskStatus OnUpdate()
        {
            if (!m_Agent.pathPending && m_Agent.remainingDistance < 0.5f)
                return TaskStatus.COMPLETED;

            return TaskStatus.RUNNING;
        }

        /// <summary>Abort method of MoveToPosition.</summary>
        /// <remarks>When the task is aborted, it stops the navAgentMesh.</remarks>
        public override void OnAbort()
        {
#if UNITY_5_6_OR_NEWER
            if (m_Agent != null)
                m_Agent.isStopped = true;
#else
            if (navAgent != null)
                navAgent.Stop();
#endif
        }
    }
}
