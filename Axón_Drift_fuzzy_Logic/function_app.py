import json
import os
import skfuzzy as fuzz
import numpy as np
from azure.cosmos import CosmosClient
import azure.functions as func

app = func.FunctionApp()

@app.function_name(name="CalculateRisk")
@app.route(route="calculate-risk", methods=["GET", "POST"])
def calculate_risk(req: func.HttpRequest) -> func.HttpResponse:
    """
    Función de Azure para calcular el riesgo de colisión usando lógica difusa.
    Entradas: distancia (km), velocidad (km/s) desde la solicitud; confianza desde Cosmos DB.
    Salida: JSON con risk_score, risk_level y confidence.
    """
    try:
        # Validar variables de entorno
        try:
            endpoint = os.environ["COSMOS_ENDPOINT"]
            key = os.environ["COSMOS_KEY"]
        except KeyError as e:
            return func.HttpResponse(
                json.dumps({"error": f"Variable de entorno faltante: {str(e)}"}),
                status_code=500,
                mimetype="application/json",
                headers={"Content-Type": "application/json"}
            )

        # Procesar datos de la solicitud
        if req.method == "POST":
            try:
                data = req.get_json()
            except ValueError:
                return func.HttpResponse(
                    json.dumps({"error": "JSON inválido en el cuerpo de la solicitud"}),
                    status_code=400,
                    mimetype="application/json",
                    headers={"Content-Type": "application/json"}
                )
            distance = float(data.get('distance', 1000))
            velocity = float(data.get('velocity', 7))
        else:  # GET
            try:
                distance = float(req.params.get('distance', 1000))
                velocity = float(req.params.get('velocity', 7))
            except (ValueError, TypeError):
                return func.HttpResponse(
                    json.dumps({"error": "Distancia y velocidad deben ser números válidos"}),
                    status_code=400,
                    mimetype="application/json",
                    headers={"Content-Type": "application/json"}
                )

        # Validar rangos de entrada
        if not (0 <= distance <= 2000):
            return func.HttpResponse(
                json.dumps({"error": "La distancia debe estar entre 0 y 2000 km"}),
                status_code=400,
                mimetype="application/json",
                headers={"Content-Type": "application/json"}
            )
        if not (0 <= velocity <= 15):
            return func.HttpResponse(
                json.dumps({"error": "La velocidad debe estar entre 0 y 15 km/s"}),
                status_code=400,
                mimetype="application/json",
                headers={"Content-Type": "application/json"}
            )

        # Conectar a Cosmos DB
        try:
            client = CosmosClient(endpoint, key)
            container = client.get_database_client("Predictions").get_container_client("Predictions")
            query = "SELECT TOP 1 * FROM c ORDER BY c.id DESC"
            items = list(container.query_items(query, enable_cross_partition_query=True))
            print(f"Items encontrados: {items}")  # Depuración
            prediction = items[0]['predictions'][0] if items and items[0].get('predictions') else {'confidence': 0.5}
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"error": f"Error en Cosmos DB: {str(e)}"}),
                status_code=500,
                mimetype="application/json",
                headers={"Content-Type": "application/json"}
            )

        # Configurar lógica difusa
        dist_range = np.arange(0, 2000, 10)
        vel_range = np.arange(0, 15, 0.5)
        conf_range = np.arange(0, 1.1, 0.1)
        risk_range = np.arange(0, 100, 1)

        # Funciones de pertenencia
        dist_near = fuzz.trimf(dist_range, [0, 0, 500])
        dist_mid = fuzz.trimf(dist_range, [250, 750, 1250])
        dist_far = fuzz.trimf(dist_range, [1000, 2000, 2000])
        vel_low = fuzz.trimf(vel_range, [0, 0, 5])
        vel_high = fuzz.trimf(vel_range, [3, 15, 15])
        conf_low = fuzz.trimf(conf_range, [0, 0, 0.5])
        conf_high = fuzz.trimf(conf_range, [0.3, 1, 1])
        risk_low = fuzz.trimf(risk_range, [0, 0, 50])
        risk_high = fuzz.trimf(risk_range, [50, 100, 100])

        # Evaluar entradas
        dist_level_near = fuzz.interp_membership(dist_range, dist_near, distance)
        dist_level_mid = fuzz.interp_membership(dist_range, dist_mid, distance)
        dist_level_far = fuzz.interp_membership(dist_range, dist_far, distance)
        vel_level_low = fuzz.interp_membership(vel_range, vel_low, velocity)
        vel_level_high = fuzz.interp_membership(vel_range, vel_high, velocity)
        conf_level_low = fuzz.interp_membership(conf_range, conf_low, prediction['confidence'])
        conf_level_high = fuzz.interp_membership(conf_range, conf_high, prediction['confidence'])

        # Reglas difusas
        rule1 = np.fmin(np.fmin(dist_level_near, vel_level_high), conf_level_high)  # Alto riesgo
        rule2 = np.fmin(dist_level_mid, vel_level_low)                              # Bajo riesgo
        rule3 = dist_level_far                                                     # Bajo riesgo
        rule4 = conf_level_low                                                     # Bajo riesgo

        # Agregar resultados
        risk_activation_high = np.fmin(rule1, risk_high)
        risk_activation_low = np.fmin(np.fmax(np.fmax(rule2, rule3), rule4), risk_low)
        aggregated = np.fmax(risk_activation_high, risk_activation_low)

        # Desfuzificar
        risk_score = fuzz.defuzz(risk_range, aggregated, 'centroid')

        # Preparar respuesta
        response = {
            'risk_score': round(risk_score, 2),
            'risk_level': 'alto' if risk_score > 80 else 'bajo',
            'confidence': round(prediction['confidence'], 4),
            'distance': distance,
            'velocity': velocity
        }

        return func.HttpResponse(
            json.dumps(response),
            status_code=200,
            mimetype="application/json",
            headers={"Content-Type": "application/json"}
        )

    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": f"Error del servidor: {str(e)}"}),
            status_code=500,
            mimetype="application/json",
            headers={"Content-Type": "application/json"}
        )