from setuptools import setup, find_packages

setup(
    name='open_llm_benchmark',
    version='0.1.0',
    author='ZHIRUI ZHOU',
    author_email='evilpsycho42@gmail.com',
    description='Evaluate the capability of open-source LLMs in Agent, formatted output, instruction following, long context retrieval, multilingual, coding, math and custom task.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/EvilPsyCHo/Open-LLM-Benchmark',
    packages=find_packages(),
    include_package_data=True,  # This includes non-code files specified in MANIFEST.in
    install_requires=[
        x for x in open("./requirements.txt", "r+").readlines() if x.strip()
    ],
    python_requires='>=3.8.4',
    classifiers=[],
    entry_points={
        'console_scripts': [
            'benchmark.retrieval = open_llm_benchmark.task.retrieval:main',
            'benchmark.format_output = open_llm_benchmark.task.format_output:main',
            'benchmark.agent = open_llm_benchmark.task.agent:main',
        ],
    },
)