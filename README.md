# Agente de Vigilância por Visão Computacional

Agente local que monitora **dezenas de câmeras RTSP** de um circuito de CFTV de
supermercado e dispara alertas de comportamento por **visão computacional**, usando
**YOLOv8** (detecção + estimativa de pose) sobre fluxos de vídeo em tempo real.

Roda localmente, na mesma rede do DVR — **sem nuvem, sem enviar imagem para fora**.

> ⚠️ **PC muito fraco pode não aguentar.** Como o agente analisa uma câmera por vez, em
> rodízio, não precisa de máquina top — mas roda dois modelos de IA (detecção e pose) o
> tempo todo. Veja [Requisitos de hardware](#requisitos-de-hardware).

> Projeto real. Versão **sanitizada** para portfólio: credenciais, IPs de DVR e
> gravações reais foram removidos. As câmeras são configuradas por variáveis de ambiente.

---

## O que ele detecta

1. **Uso de celular por funcionário** — pessoa de uniforme usando o celular por tempo
   acumulado (ex.: 15 s dentro de uma janela de 50 s por câmera).
2. **Consumo de mercadoria na loja** — em zonas de hortifrúti/laticínios, pessoa com a
   mão perto da boca por tempo acumulado, indício de consumo não pago.

Ao disparar, salva o **frame anotado**, um **clipe de vídeo** do momento e registra tudo
em log — com *cooldown* para não repetir o mesmo alerta.

## O pipeline de visão computacional

```
DVR (RTSP) ──▶ OpenCV/FFmpeg ──▶ pré-processamento ──▶ YOLOv8 ──▶ regras temporais ──▶ alerta
   32+18        captura de       CLAHE (contraste)    detecção    acúmulo por janela    frame+vídeo
   canais       frame              + resize            + pose      com decaimento          + log
```

- **RTSP + FFmpeg**: cada câmera é um fluxo `rtsp://.../cam/realmonitor?channel=N`.
  A captura usa o backend FFmpeg do OpenCV com transporte TCP (mais estável que UDP em
  rede saturada). O protótipo das câmeras foi validado direto com
  `ffplay -rtsp_transport tcp`.
- **Multiplexação (rodízio de câmeras)**: decisão de projeto tomada por limitação de
  hardware — na operação real havia ~50 canais e **um PC só, sem GPU**, que não aguentava
  analisar tudo ao mesmo tempo. Por isso o agente analisa **um grupo de câmera por vez**,
  por um intervalo fixo, em rodízio — e roda a inferência de IA a cada ~1 s, não a cada
  frame. É isso que permite rodar em hardware modesto.
- **YOLOv8 detecção** (`yolov8n`/`yolov8s`): localiza pessoas e celulares.
- **YOLOv8 pose** (`yolov8n-pose`): keypoints de mão e rosto para o gesto mão-boca.
- **Pré-processamento CLAHE**: equalização de contraste antes do YOLO, porque imagem de
  CFTV costuma ser escura e granulada.
- **Identificação de uniforme por cor (HSV)**: máscara de azul/branco no tronco para
  separar funcionário de cliente, com suporte a zonas (polígonos) por câmera.
- **Regras temporais**: cada suspeita acumula "tempo de evidência" e **decai** quando a
  evidência some, evitando falsos positivos de frames isolados.

## Robustez (o que faz funcionar 24/7)

- **Reinício automático** de câmera quando o RTSP trava ou fica sem frame por X segundos.
- **Timeout de RTSP** e reinício do processo em erro fatal, para sobreviver a quedas de rede.
- **Detecção de movimento** barata (proporção de pixels alterados) como *gate* antes de
  acionar a IA — não gasta inferência em cena parada.
- Parâmetros de sensibilidade todos no topo do arquivo, documentados.

## Stack

`Python` · `OpenCV` · `FFmpeg/RTSP` · `Ultralytics YOLOv8` (detect + pose) · `NumPy` ·
threads (`ThreadPoolExecutor`) para captura concorrente

## Requisitos de hardware

O hardware necessário depende de **como** você vai usar (o rodízio existe justamente para
rodar em hardware limitado — veja *Multiplexação* acima):

- **Rodízio (padrão, `CAMERAS_GRUPO = 1`)** — analisa uma câmera por vez, alternando entre
  elas. Roda em um PC intermediário, só com CPU. Só não aguenta máquina **muito** fraca
  (Celeron, Atom, dual-core antigo, 4 GB de RAM): a IA atrasa, as câmeras reiniciam por
  timeout e os alertas não saem.
- **Várias câmeras ao mesmo tempo (`CAMERAS_GRUPO > 1`)** — a carga se multiplica pelo
  número de câmeras simultâneas. **Aqui é preciso investir em hardware**, principalmente
  em uma GPU NVIDIA.

| | Rodízio (padrão) | Várias câmeras simultâneas |
|---|---|---|
| **CPU** | 4+ núcleos (Core i5 / Ryzen 5 ou equivalente) | 8+ núcleos (Core i7 / Ryzen 7 ou superior) |
| **RAM** | 8 GB (16 GB recomendado) | 16–32 GB |
| **GPU** | não precisa | **NVIDIA com CUDA** (ex.: RTX 3060 ou superior) |
| **Rede** | cabo, mesma LAN do DVR (evite Wi-Fi) | Gigabit cabeado |
| **Disco** | espaço livre para os clipes de alerta | SSD |

- Com GPU NVIDIA + CUDA, o Ultralytics usa a placa automaticamente.
- Se estiver lento, aumente `IA_INTERVALO_SEG` ou reduza `CAMERAS_GRUPO`.
- Os valores acima são orientação, não benchmark formal.
- Também é necessário um DVR/NVR com RTSP habilitado e acesso de rede a ele.

## Como rodar

```bash
pip install -r requirements.txt
cp .env.example .env     # configure os fluxos RTSP das câmeras
python vigilancia.py     # os pesos YOLOv8 são baixados na primeira execução
```

No Windows há os lançadores `instalar.bat` / `executar.bat` para uso por pessoas não
técnicas (duplo clique), parte do empacotamento entregue ao cliente.

## Configuração (topo de `vigilancia.py`)

| Parâmetro | Função |
|---|---|
| `CONF_PESSOA`, `CONF_CELULAR` | confiança mínima do YOLO |
| `TEMPO_CELULAR_ALERTA`, `TEMPO_CONSUMO_ALERTA` | tempo acumulado para disparar |
| `AREAS_CONSUMO_MERCADORIA` | zonas (por canal) onde faz sentido procurar consumo |
| `CAMERAS_RESTRITAS_BRANCO` / `ZONAS_RESTRITAS_BRANCO` | onde o uniforme conta |
| `COOLDOWN_ALERTA_SEG` | janela de silêncio após um alerta |

---

*Autoria: cyberpunk. Código sanitizado para fins de portfólio.*

**Suporte:** repositório de portfólio, sem suporte de instalação. Problemas de
desempenho em hardware abaixo do indicado para o seu modo de uso não serão tratados como bug.
