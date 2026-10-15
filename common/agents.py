"""명시적으로 선택한 연구 구조를 실행한다. 질문으로 구조를 자동 선택하지 않는다."""
from importlib import import_module
from common.evaluation.runner import question_inputs

ARCHITECTURES = {
    'single_agent': '단일 에이전트',
    'mas_central_parallel': '중앙 조정·병렬 수행형',
    'mas_sequential': '순차 근거전달형',
    'mas_decentralized': '분산 상호협력형',
    'mas_hierarchical': '계층적 조정형',
    'mas_review': '협력 후검토형',
    'mas_dynamic': '동적 작업조정형',
}


class MultiAgent:
    def __init__(self, architecture, settings, *, tools=None, model_factory=None):
        if architecture not in ARCHITECTURES or architecture == 'single_agent':
            raise ValueError('지원하는 멀티에이전트 구조를 지정하세요.')
        self.architecture = architecture
        self.settings = settings
        self.tools = tools
        self.model_factory = model_factory

    def answer(self, inputs):
        from common.orchestration import Context
        clean = question_inputs(inputs)
        question = clean['messages'][0][1]
        ctx = Context(self.settings, tools=self.tools, model_factory=self.model_factory)
        builder = import_module(self.architecture + '.graph').build_graph
        initial = {'question': question, 'tasks': [], 'reports': {}, 'events': [],
                   'final_answer': '', 'status': 'running', 'stop_reason': ''}
        try:
            result = builder(ctx).invoke(initial, config={
                'recursion_limit': 100,
                'run_name': self.architecture + '_qa',
                'tags': [self.architecture, 'sbiz'],
                'metadata': {'architecture': self.architecture,
                             'model': self.settings.upstage_model,
                             'experiment_date': str(self.settings.experiment_date)},
            })
        except Exception as exc:
            # SDK 원문·비밀값을 최종 결과나 trace 예외로 다시 전파하지 않는다.
            result = {**initial, 'status': 'error', 'stop_reason': 'graph_error:' + type(exc).__name__}
            if hasattr(ctx, 'reports_snapshot'):
                result['reports'] = ctx.reports_snapshot()
        result['metrics'] = ctx.metrics()
        result['events'] = ctx.events_snapshot()
        result['architecture'] = self.architecture
        result['evidence'] = [item for report in result.get('reports', {}).values()
                              for item in report.get('evidence', [])]
        return result


def create_agent(architecture, settings, **kwargs):
    if architecture == 'single_agent':
        from single_agent.service import SingleAgent
        return SingleAgent(settings, **kwargs)
    return MultiAgent(architecture, settings, **kwargs)
