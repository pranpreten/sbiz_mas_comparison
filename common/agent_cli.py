"""모든 연구 구조의 질문 한 개 실행 진입점."""
import argparse
import json
from pathlib import Path
from common.agents import ARCHITECTURES, create_agent
from common.config import load_settings


def main(architecture, argv=None):
    parser = argparse.ArgumentParser(description=ARCHITECTURES[architecture] + ' 질문 실행')
    parser.add_argument('question', nargs='?')
    parser.add_argument('--env', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    question = args.question or input('질문: ').strip()
    if not question.strip():
        parser.error('질문을 입력해 주세요.')
    result = create_agent(architecture, load_settings(args.env)).answer({'messages': [['user', question]]})
    print(result.get('final_answer') or '최종 답변을 완료하지 못했습니다.')
    print('상태:', result['status'], '/', result['stop_reason'])
    print(json.dumps(result['metrics'], ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({k:v for k,v in result.items() if k != 'messages'},
                              ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    return int(result['status'] == 'error')


def entrypoint(architecture):
    try:
        return main(architecture)
    except KeyboardInterrupt:
        print('사용자 요청으로 중단했습니다.')
        return 130
    except Exception as exc:
        print('실행 실패:', type(exc).__name__)
        return 1
