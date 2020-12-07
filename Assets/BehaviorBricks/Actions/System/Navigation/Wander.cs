using Pada1.BBCore;
using Pada1.BBCore.Tasks;
using UnityEngine;

namespace BBUnity.Actions
{
    /// <summary>
    /// Makes the tank wander
    /// </summary>
    [Action("Navigation/Wander")]
    [Help("make an object wander")]
    public class Wander : GOAction
    {

        [InParam("IsWanderer")]
        public bool isWanderer;

        private UnityEngine.AI.NavMeshAgent m_Agent;
        private float radius = 1.0f;
        private float offset = 4.0f;

        public override void OnStart()
        {
            if(!isWanderer) //Only do this is it's thw wandering tank
            {
                return;
            }

            m_Agent = gameObject.GetComponent<UnityEngine.AI.NavMeshAgent>();

            if (!m_Agent.pathPending && m_Agent.remainingDistance < 0.5f)
            {
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
        }

        public override TaskStatus OnUpdate()
        {
            return TaskStatus.COMPLETED;
        }
    }
}
