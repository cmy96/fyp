html_layout = '''<!DOCTYPE html>
                    <html>
                        <head>
                            {%metas%}
                            <title>{%title%}</title>
                            {%favicon%}
                            {%css%}
                        </head>

                        <body>
                            <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
                            <a class="navbar-brand" href="#"><img src="C:\wamp64\www\\fyp2\\application\\templates\Logo.png"> C.A.R.E</a>
                              <span class="navbar-text navbar-text-color">Breast Cancer Prediction Tool</span>
                            <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                                <span class="navbar-toggler-icon"></span>
                            </button>
    
                            <div class="collapse navbar-collapse" id="navbarSupportedContent">
                            <ul class="navbar-nav ml-auto">
                                <li class="nav-item">
                                    <a class="nav-link" href="/"><i class="fas fa-calculator"></i> Calculator </a>
                                </li>
                                <li class="nav-item active">
                                    <a class="nav-link" href="/dashapp"><i class="fas fa-chart-pie"></i> Dashboard <span class="sr-only">(current)</span></a>
                                </li>
          
                            </div>
                            </nav>
                            {%app_entry%}

                            <footer>
                                {%config%}
                                {%scripts%}
                                {%renderer%}
                            </footer>
                        </body>
                    </html>'''
